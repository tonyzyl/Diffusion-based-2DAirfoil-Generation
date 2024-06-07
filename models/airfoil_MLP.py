import io
from PIL import Image
import os
import pickle
from tqdm import tqdm
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from torchvision.transforms import ToTensor


from .mlp import CFGResNet
from einops import repeat

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

class EulerMaruyama(nn.Module):
    def __init__(self, 
                 num_time_steps=500, 
                 eps=1e-7,
                 resample=1,
                 intermediate_steps=100000):
        super().__init__()

        self.num_time_steps = num_time_steps
        self.eps = eps
        self.intermediate_steps = intermediate_steps
        self.resample = resample

    @torch.no_grad()
    def predictor_step(self, x, c, t, context_mask, step_size, unet, sde, device, **kwargs):
        mean_x = x - (sde.f(x, t) - sde.g(t)**2*unet(x, c, t, context_mask, **kwargs))*step_size
        x = mean_x + torch.sqrt(step_size)*sde.g(t)*torch.randn_like(x)
        return x, mean_x

    @torch.no_grad()
    def forward(self, unet, sde, data_size, device, c=None, return_intermediates=False, **kwargs):
        batch_size = data_size[0]
        noise = sde.sample_prior(data_size, device=device)
        time_steps = torch.linspace(1., self.eps, self.num_time_steps, device=device)
        step_size = time_steps[0]-time_steps[1]

        if c is None:
            c = torch.zeros((batch_size, unet.cond_size), device=device)
        else:
            c = c[:batch_size]
        context_mask = torch.zeros_like(c)

        x = noise + 0.
        intermediates = []
        i = 1
        for time_step in tqdm(time_steps, desc="Sampling", unit="iteration"):
            batch_time_step = torch.ones(batch_size, device=device) * time_step
            x, mean_x = self.predictor_step(x, c, batch_time_step, context_mask, step_size, unet, sde, device, **kwargs)
            if i % self.intermediate_steps == 0:
                intermediates.append(x)
            i += 1

        # no noise in last step
        return mean_x, intermediates

    @torch.no_grad()
    def inpaint(self, unet, sde, data_size, values, inds, device):
        # if sensor values exist
        if len(inds) > 0:
            # sample with inpainting
            batch_size = data_size[0]
            noise = sde.sample_prior(data_size, device=device)
            time_steps = torch.linspace(1., self.eps, self.num_time_steps, device=device)
            step_size = time_steps[0]-time_steps[1]
            c = torch.zeros((batch_size, unet.cond_size), device=device)
            context_mask = torch.zeros_like(c)
            x = noise + 0.
            intermediates = []
            i = 0

            # normalize the values if needed
            normalized_values = values + 0. #values*self.residual.sigma_p + self.residual.mu_p

            # initial replacement of data values with noisy known values
            noisy_values, _, _ = sde.forward(normalized_values, torch.ones(batch_size, device=device))
            #noisy_values = normalized_values + 0.
            x[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = noisy_values[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]

            for time_step in tqdm(time_steps, desc="Sampling", unit="iteration"):
                batch_time_step = torch.ones(batch_size, device=device) * time_step


                for i in range(self.resample):
                    # predict backward in time for one step
                    x, mean_x = self.predictor_step(x, c, batch_time_step, context_mask, step_size, unet, sde, device)

                    # replace x with noisy versions of the known dimensions
                    noisy_values, _, _ = sde.forward(normalized_values, batch_time_step-step_size)
                    #noisy_values = normalized_values + 0.
                    x[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = noisy_values[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]

                    if i < (self.resample-1):
                        # simulate forward in time for one step
                        x = x + sde.f(x, batch_time_step)*step_size + self.g(batch_time_step)*torch.sqrt(step_size)*torch.randn_like(x)

                if i % self.intermediate_steps == 0:
                    intermediates.append(x)
                i += 1

            mean_x[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]] = normalized_values[inds[:, 0], inds[:, 1], inds[:, 2], inds[:, 3]]

            return mean_x

        else:
            mean_x, _ = self.forward(unet, sde, data_size, device)
            return mean_x

class ProbabilityFlowODE(EulerMaruyama):
    def __init__(self,
                 num_time_steps=500, 
                 eps=1e-7,
                 intermediate_steps=100000):
        super().__init__(num_time_steps, eps, intermediate_steps)

    @torch.no_grad()
    def predictor_step(self, x, c, t, context_mask, step_size, unet, sde, device):
        # predictor step has a slightly modified form and no noise
        mean_x = x - (sde.f(x, t) - 0.5*sde.g(t)**2*unet(x, c, t, context_mask))*step_size
        x = mean_x + 0.
        return x, mean_x

class EDMLoss:
    # 1D EDM Loss
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None):
        rnd_normal = torch.randn([images.shape[0], 1], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

def edm_sampler(
    net, latents, class_labels=None, randn_like=torch.randn_like,
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=0,
    deterministic=False
):

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    x_next = latents.to(torch.float64) * t_steps[0]

    whole_trajectory = torch.zeros((num_steps, *x_next.shape), dtype=torch.float64)
    # Main sampling loop.
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1

        x_cur = x_next
        if not deterministic:
            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)
        else:
            t_hat = t_cur
            x_hat = x_cur
        # Euler step.
        denoised = net(x_hat, repeat(t_hat.reshape(-1), 'w -> h w', h=x_hat.shape[0]), class_labels).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        # Apply 2nd order correction.
        if i < num_steps - 1:
            denoised = net(x_next, repeat(t_next.reshape(-1), 'w -> h w', h=x_next.shape[0]), class_labels).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

        whole_trajectory[i] = x_next

    return x_next, whole_trajectory

class EDM_CFG(torch.nn.Module):
    def __init__(self,
        in_dim, out_dim, cond_size,
        model_dim      = 128,      # dim multiplier.
        dim_mult        = [1,1,1,1],# dim multiplier for each resblock layer.
        dim_mult_emb    = 4,
        num_blocks          = 4,        # Number of resblocks(mid) per level.
        dropout             = 0.,      # Dropout rate.
        emb_type            = "sinusoidal",# Timestep embedding type
        dim_mult_time  = 1,        # Time embedding size
        use_fp16        = False,            # Execute the underlying model at FP16 precision?
        sigma_min       = 0,                # Minimum supported noise level.
        sigma_max       = float('inf'),     # Maximum supported noise level.
        sigma_data      = 0.5,              # Expected standard deviation of the training data.
        model_type      = 'CFGResNet',   # Class name of the underlying model.
        **model_kwargs,                     # Keyword arguments for the underlying model.
    ):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.label_dim = cond_size
        self.use_fp16 = use_fp16
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        ###########
        self.model = globals()[model_type](self.in_dim, self.out_dim, self.label_dim, model_dim=model_dim, dim_mult=dim_mult, dim_mult_emb=dim_mult_emb, num_blocks=num_blocks,
                                           dropout=dropout, emb_type=emb_type, dim_mult_time=dim_mult_time, **model_kwargs)

    def forward(self, x, sigma, class_labels=None, force_fp32=False,  **model_kwargs):

        x = x.to(torch.float32)

        sigma = sigma.to(torch.float32).reshape(-1, 1)
        class_labels = None if (self.label_dim == 0 or class_labels is None) else class_labels.to(torch.float32).reshape(-1, self.label_dim)
        dtype = torch.float16 if (self.use_fp16 and not force_fp32 and x.device.type == 'cuda') else torch.float32

        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4

        x_in = c_in * x
        F_x = self.model((x_in).to(dtype), class_labels, c_noise.flatten(), **model_kwargs)

        assert F_x.dtype == dtype
        D_x = c_skip * x + c_out * F_x.to(torch.float32)
        return D_x

    def round_sigma(self, sigma):
        return torch.as_tensor(sigma)

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]
        self.seeds = seeds
        self.device = device

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

# Dataset Class
class Airfoil1D_Dataset(Dataset):
    def __init__(self, unique_y_coords, y_coord_mapping, processed_cond_data, index_to_name_mapping):
        self.unique_y_coords = unique_y_coords
        self.y_coord_mapping = y_coord_mapping
        self.processed_cond_data = processed_cond_data
        self.index_to_name_mapping = index_to_name_mapping
    
    def __len__(self):
        return len(self.processed_cond_data)

    def __getitem__(self, idx):
        af_name = self.index_to_name_mapping[idx]
        y_coord_idx = self.y_coord_mapping[af_name]
        y_coord = self.unique_y_coords[y_coord_idx]
        cond_data = self.processed_cond_data[idx]

        return torch.tensor(y_coord, dtype=torch.float32), torch.tensor(cond_data, dtype=torch.float32)

class VP_1D(nn.Module):
    def __init__(self, 
                 beta_min=1e-4, 
                 beta_max=1.0):
        super().__init__()

        self.beta_min = beta_min
        self.beta_max = beta_max

    def beta(self, t):
        return self.beta_min + t*(self.beta_max - self.beta_min)

    def eps(self, noise, score, t, r):
        # used for corrector step in predictor-corrector methods
        alpha = 1 - self.beta(t)
        noise_norm = torch.linalg.norm(noise.reshape((noise.shape[0], -1)), dim=-1).mean()
        score_norm = torch.linalg.norm(score.reshape((score.shape[0], -1)), dim=-1).mean()
        eps = 2*alpha*(r*noise_norm/score_norm)**2
        return eps[:, None]

    def f(self, x, t):
        return -0.5*self.beta(t)[:, None]*x

    def g(self, t):
        return torch.sqrt(self.beta(t))[:, None]

    def q_mu(self, x0, t):
        return x0*torch.exp(-0.25*t**2*(self.beta_max-self.beta_min)-0.5*t*self.beta_min)[:, None]

    def q_std(self, x0, t):
        return torch.sqrt(1-torch.exp(-0.5*t**2*(self.beta_max-self.beta_min)-t*self.beta_min))[:, None]

    def forward(self, x0, t):
        # forward SDE transition kernel. Return noisy sample, noise, and standard deviation
        z = torch.randn_like(x0)
        std = self.q_std(x0, t)
        mu = self.q_mu(x0, t)
        return mu + z*std, std, z

    def sample_prior(self, data_size, device):
        return torch.randn(data_size, device=device)
    
    def prior_logp(self, z):
        shape = z.shape
        N = np.prod(shape[1:])
        logps = -N / 2. * np.log(2 * np.pi) - torch.sum(z ** 2, dim=(1)) / 2.
        return logps

def p_losses_cond(score_model, sde, x0, c, **kwargs):
    # compute losses (including ELBO, score-matching loss)
    t = torch.rand(x0.shape[0], device=x0.device) * (1. - 1e-5) + 1e-5
    x_perturbed, std, z = sde.forward(x0, t)
    context_mask = torch.zeros_like(c)
    score_pred = score_model(x_perturbed, c, t, context_mask, **kwargs)

    loss_dict = {}

    log_prefix = 'train' 

    # score-mathcing objective function
    score_loss = torch.sum((score_pred*std + z)**2, dim=(1))

    loss_dict.update({f'{log_prefix}/loss_score': score_loss.mean()})

    lamb = sde.g(t)**2
    loss_vlb = lamb*score_loss
    loss_vlb = loss_vlb.mean()
    loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

    loss = score_loss.mean()

    loss_dict.update({f'{log_prefix}/loss': loss})

    return loss, loss_dict

def plot_to_tensor(fig):
    """Convert a Matplotlib figure to a 3D tensor for TensorBoard."""
    # Save the plot to a PNG in memory
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    
    # Convert PNG buffer to PIL image
    pil_img = Image.open(buf)
    
    # Convert PIL image to tensor
    return ToTensor()(pil_img)

if __name__ == '__main__':
    # Load data from disk
    with open('./pickle_files/unique_y_coords.pkl', 'rb') as f:
        unique_y_coords = pickle.load(f)

    with open('./pickle_files/y_coord_mapping.pkl', 'rb') as f:
        y_coord_mapping = pickle.load(f)

    with open('./pickle_files/index_to_name_mapping.pkl', 'rb') as f:
        index_to_name_mapping = pickle.load(f)

    with open('./pickle_files/processed_cond_data.pkl', 'rb') as f:
        processed_cond_data = pickle.load(f)

    with open('./pickle_files/minmax_scaler.pkl', 'rb') as f:
        minmax_scaler = pickle.load(f)

    with open('./pickle_files/pca.pkl', 'rb') as f:
        pca = pickle.load(f)

    seed_everything(0)

    #dataset = Airfoil2D_Dataset(airfoil_df, airfoil_coord_df)
    unique_y_coords = pca.transform(np.asarray(unique_y_coords).reshape(-1, 200))
    dataset = Airfoil1D_Dataset(unique_y_coords, y_coord_mapping, processed_cond_data, index_to_name_mapping)
    generator = torch.Generator().manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8,0.2], generator=generator)

    # optimization
    learning_rate = 1E-4
    num_epochs = 500
    num_components = 6

    beta_min = 1e-4 # for SDE noise [VP]
    beta_max = 1 # for SDE noise [VP]
    cond_scale =1
    rescaled_phi = 0

    device='cuda'
    last_loss_values = []
    fid_values = []
    avg_nll_values = []
    KLD_arr = []
    H_gen_arr = []
    H_true_arr = []

    Training = True
    method = 'EDM' # 'EDM' or 'CFG'
    save_path = './mdl_weight/edm.pth' 

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True
    )


    vp = VP_1D(beta_min, beta_max)

    if method == 'CFG':
        print('Using CFG')
        model = CFGResNet(num_components, num_components, cond_size=4, model_dim=128,
                        dim_mult=[1,2,2], dim_mult_emb=4, num_blocks=2,
                        dropout=0, emb_type="sinusoidal", dim_mult_time=1, 
                        dim_mult_cond=1, cond_drop_prob=0, adaptive_scale=True, skip_scale=1.0, affine=True)
    elif method == 'EDM':
        print('Using EDM')
        model = EDM_CFG(num_components, num_components, cond_size=4, model_dim=128,
                        dim_mult=[1,2,2], dim_mult_emb=4, num_blocks=2,
                        dropout=0, emb_type="sinusoidal", dim_mult_time=1, 
                        dim_mult_cond=1, cond_drop_prob=0, adaptive_scale=True, skip_scale=1.0, affine=True)
        loss_fn = EDMLoss()
    else:
        raise NotImplementedError

    if Training:
        log_dir = './tensor_log_MLP/edm/'
        writer = SummaryWriter(log_dir)
        data_log_interval = 10
        data_sample_size = 100
        grid_size = 10
        if method == 'CFG':
            pf_sampler = ProbabilityFlowODE(1000, 1e-7)

        model.train()
        model.to(device)

        '''
        optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate,
            )
        '''
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1)
        scheduler_iters = len(train_loader)

        global_step = 0
        frames = []
        loss_v = []
        loss_avg = []
        #val_loss_avg = []
        best_val_loss = float('inf')
        print('training ...')
        for epoch in trange(num_epochs):
            model.train()
            train_loss = 0.
            num_items = 0

            for step, batch in enumerate(train_loader):
                x = batch[0]
                c = batch[1]
                x = x.to(device)
                c = c.to(device)
                if method == 'CFG':
                    loss,_ = p_losses_cond(model, vp, x, c)
                elif method == 'EDM':
                    tmp_loss = loss_fn(model, x, c)
                    loss = tmp_loss.sum().mul(1/x.shape[0])
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step(epoch + step / scheduler_iters)
                train_loss += loss.item() * x.shape[0]
                num_items += x.shape[0]
                loss_v.append(loss.item())
                loss_avg.append(train_loss / num_items)
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    x_val = batch[0]
                    c_val = batch[1]
                    x_val = x_val.to(device)
                    c_val = c_val.to(device)
                    if method == 'CFG':
                        loss,_ = p_losses_cond(model, vp, x, c)
                    elif method == 'EDM':
                        tmp_loss = loss_fn(model, x, c)
                        loss = tmp_loss.sum().mul(1/x.shape[0])
                    val_loss += loss.item() * x_val.size(0)

            val_loss /= len(val_loader.dataset)

            # Save the model if validation loss has decreased
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
            # Optionally print epoch statistics
            print(f'Epoch {epoch}: Training Loss: {train_loss / len(train_loader.dataset):.4f}, Validation Loss: {val_loss:.4f}')

            writer.add_scalar('Loss/train', train_loss / num_items, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], epoch)
            if (epoch+1) % data_log_interval == 0:
                #if epoch >= 19:
                #    torch.save(model.state_dict(), 'mdl_weight/reduced/reduced_epoch_{}.pth'.format(epoch))
                with torch.no_grad():
                    model.eval()

                    # Sample a subset from train dataset
                    train_input, train_cond = next(iter(train_loader))
                    train_input = train_input.to(device)[:data_sample_size]  # Take a subset for visualization
                    train_cond = train_cond.to(device)[:data_sample_size]

                    # Sample a subset from validation dataset
                    val_input, val_cond = next(iter(val_loader))
                    val_input = val_input.to(device)[:data_sample_size]  # Take a subset for visualization
                    val_cond = val_cond.to(device)[:data_sample_size]

                    # Generate samples
                    if method == 'CFG':
                        train_samples,_ = pf_sampler(model, vp, train_input.shape, device, c=train_cond)
                        val_samples,_ = pf_sampler(model, vp, val_input.shape, device, c=val_cond)
                    elif method == 'EDM':
                        rnd = StackedRandomGenerator(device, range(train_input.shape[0]))
                        train_latents = rnd.randn([train_input.shape[0], model.in_dim],device=device)
                        val_latents = rnd.randn([val_input.shape[0], model.in_dim],device=device)
                        with torch.no_grad():
                            train_samples, _ = edm_sampler(model, latents=train_latents, class_labels=train_cond, randn_like=rnd.randn_like)
                            val_samples, _ = edm_sampler(model, latents=val_latents, class_labels=val_cond, randn_like=rnd.randn_like)

                    train_samples = pca.inverse_transform(train_samples[:].cpu().detach().numpy()).reshape(-1,2,100)
                    val_samples = pca.inverse_transform(val_samples[:].cpu().detach().numpy()).reshape(-1,2,100)
                    train_input = pca.inverse_transform(train_input[:].cpu().detach().numpy()).reshape(-1,2,100)
                    val_input = pca.inverse_transform(val_input[:].cpu().detach().numpy()).reshape(-1,2,100)

                    train_images = []
                    for i in range(data_sample_size):
                        plt.figure(figsize=(2, 2))
                        plt.plot(train_input[i,0,:])
                        plt.plot(train_input[i,1,:])
                        plt.plot(train_samples[i,0,:])
                        plt.plot(train_samples[i,1,:])
                        plt.xticks([])
                        fig = plt.gcf()
                        train_images.append(plot_to_tensor(fig))

                    # Combine images into a grid and log to TensorBoard
                    train_grid = make_grid(train_images, nrow=grid_size)
                    writer.add_image('Train_Samples_Grid', train_grid, epoch)

                    # Repeat the process for validation samples
                    val_images = []
                    for i in range(data_sample_size):
                        plt.figure(figsize=(2, 2))
                        plt.plot(val_input[i,0,:])
                        plt.plot(val_input[i,1,:])
                        plt.plot(val_samples[i,0,:])
                        plt.plot(val_samples[i,1,:])
                        plt.xticks([])
                        fig = plt.gcf()
                        val_images.append(plot_to_tensor(fig))

                    val_grid = make_grid(val_images, nrow=grid_size)
                    writer.add_image('Validation_Samples_Grid', val_grid, epoch)

        writer.close()
    else:
        model.load_state_dict(torch.load(save_path))
        model.to(device)