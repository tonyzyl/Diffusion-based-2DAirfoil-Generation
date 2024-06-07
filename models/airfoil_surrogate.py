import os
import pickle
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import Dataset

class Affine(nn.Module):
    #https://github.com/facebookresearch/deit/blob/263a3fcafc2bf17885a4af62e6030552f346dc71/resmlp_models.py#L16C9-L16C9
    def __init__(self, dim):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.alpha * x + self.beta  

class ResNetBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0,
                 skip_scale=1, adaptive_scale=True, affine=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.res_linear = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

        if affine:
            self.pre_norm = Affine(in_dim)
            self.post_norm = Affine(out_dim)
        else:
            self.pre_norm = nn.Identity()
            self.post_norm = nn.Identity()

    def forward(self, x):
        #print(x.shape, emb.shape)
        orig = x
        x = self.pre_norm(x)
        x = self.linear1(nn.functional.silu(x))
        x = self.linear2(nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = self.post_norm(x)
        x = x.add_(self.res_linear(orig))
        x = x * self.skip_scale

        return x

class ResNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim,
                model_dim      = 128,      # dim multiplier.
                dim_mult        = [1,1,1,1],# dim multiplier for each resblock layer.
                num_blocks          = 4,        # Number of resblocks(mid) per level.
                dropout             = 0.,      # Dropout rate.
                adaptive_scale  = True,     # Feature-wise transformations, FiLM
                skip_scale      = 1.0,      # Skip connection scaling
                affine          = False    # Affine normalization for MLP
                ):

        super().__init__()

        block_kwargs = dict(dropout = dropout, skip_scale=skip_scale, adaptive_scale=adaptive_scale, affine=affine)


        self.first_layer = nn.Linear(in_dim, model_dim)
        self.blocks = nn.ModuleList()
        cout = model_dim
        for level, mult in enumerate(dim_mult):
            for _ in range(num_blocks):
                cin = cout
                cout = model_dim * mult
                self.blocks.append(ResNetBlock(cin, cout, **block_kwargs))
        self.final_layer = nn.Linear(cout, out_dim)

    def forward(self, x):
        # Mapping
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_layer(nn.functional.silu(x))
        return x

# Dataset Class
class Airfoil_surrogate_Dataset(Dataset):
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
        y_coord = torch.tensor(y_coord, dtype=torch.float32)
        cond_data = torch.tensor(cond_data, dtype=torch.float32)

        return torch.cat((y_coord, cond_data[2:])), cond_data[:2]

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

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
    dataset = Airfoil_surrogate_Dataset(unique_y_coords, y_coord_mapping, processed_cond_data, index_to_name_mapping)
    generator = torch.Generator().manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dataset, [0.8,0.2], generator=generator)

    # optimization
    learning_rate = 1E-3
    num_epochs = 1000
    num_components = 6

    device='cuda'

    Training = True

    batch_size = 128
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=batch_size, shuffle=True
    )

    model = ResNet(in_dim = num_components+2, out_dim=2, model_dim=128, dim_mult=[1,1,1], num_blocks=1, 
                   dropout=0., adaptive_scale=True, skip_scale=1.0, affine=False)

    model.train()
    model.to(device)

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
        )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6, last_epoch=-1)
    scheduler_iters = len(train_loader)

    best_val_loss = float('inf')
    save_path = './mdl_weight/surrogate_model.pth' 
    # Training loop
    for epoch in trange(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            y_hat = model(x)
            loss = torch.nn.functional.mse_loss(y_hat, y)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + batch_idx / scheduler_iters)
        train_loss /= len(train_loader)
        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)
                y_hat = model(x)
                val_loss += torch.nn.functional.mse_loss(y_hat, y)
            val_loss /= len(val_loader)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), save_path)
        if epoch % 10 == 0:
            print(f'Epoch {epoch}: train_loss={train_loss:.7f}, val_loss={val_loss:.7f}')