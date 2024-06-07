import torch
from torch import nn
from einops import rearrange, repeat
from functools import partial

from .embeddings import PositionalEmbedding
from .utils import zero_module

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

class CondResNetBlock(torch.nn.Module):
    def __init__(self, in_dim, out_dim, time_emb_dim, cond_emb_dim, dropout=0,
                 skip_scale=1, adaptive_scale=True, affine=False):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.emb_dim = time_emb_dim
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.res_linear = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.map_cond = nn.Linear(time_emb_dim+cond_emb_dim, out_dim*(2 if adaptive_scale else 1))

        if affine:
            self.pre_norm = Affine(in_dim)
            self.post_norm = Affine(out_dim)
        else:
            self.pre_norm = nn.Identity()
            self.post_norm = nn.Identity()

    def forward(self, x, time_emb=None, cond_emb=None):
        #print(x.shape, emb.shape)
        orig = x
        emb = torch.cat((time_emb, cond_emb), dim = -1)
        params = nn.functional.silu(self.map_cond(emb).to(x.dtype))
        x = self.pre_norm(x)
        x = self.linear1(nn.functional.silu(x))
        if self.adaptive_scale:
            scale, shift = params.chunk(2, dim=-1)
            x = nn.functional.silu(torch.addcmul(shift, x, scale+1))
        else:
            x = nn.functional.silu(x.add_(params))

        x = self.linear2(nn.functional.dropout(x, p=self.dropout, training=self.training))
        x = self.post_norm(x)
        x = x.add_(self.res_linear(orig))
        x = x * self.skip_scale

        return x


class CFGResNet(torch.nn.Module):
    # https://github.com/lucidrains/denoising-diffusion-pytorch/blob/main/denoising_diffusion_pytorch/classifier_free_guidance.py
    def __init__(self, in_dim, out_dim, cond_size,
                model_dim      = 128,      # dim multiplier.
                dim_mult        = [1,1,1,1],# dim multiplier for each resblock layer.
                dim_mult_emb    = 4,
                num_blocks          = 4,        # Number of resblocks(mid) per level.
                dropout             = 0.,      # Dropout rate.
                emb_type            = "sinusoidal",# Timestep embedding type
                dim_mult_time  = 1,        # Time embedding size
                dim_mult_cond   = 1,        # Conditional embedding size
                cond_drop_prob      = 0.0,      # Probability of using null emb
                adaptive_scale  = True,     # Feature-wise transformations, FiLM
                skip_scale      = 1.0,      # Skip connection scaling
                affine          = False    # Affine normalization for MLP
                ):

        super().__init__()

        emb_dim = model_dim * dim_mult_emb
        time_dim = model_dim * dim_mult_time
        cond_dim = model_dim * dim_mult_cond
        block_kwargs = dict(dropout = dropout, skip_scale=skip_scale, adaptive_scale=adaptive_scale, affine=affine)

        self.null_emb = nn.Parameter(torch.randn(emb_dim))
        self.cond_size = cond_size
        self.cond_drop_prob = cond_drop_prob

        self.map_time = PositionalEmbedding(size=time_dim, type=emb_type)
        self.map_cond = PositionalEmbedding(size=cond_dim, type=emb_type)
        self.map_time_layer = nn.Linear(time_dim, emb_dim)
        self.map_cond_layer = nn.Linear(cond_dim*cond_size, emb_dim)

        self.first_layer = nn.Linear(in_dim, model_dim)
        self.blocks = nn.ModuleList()
        cout = model_dim
        for level, mult in enumerate(dim_mult):
            for _ in range(num_blocks):
                cin = cout
                cout = model_dim * mult
                self.blocks.append(CondResNetBlock(cin, cout, emb_dim, emb_dim, **block_kwargs))
        self.final_layer = nn.Linear(cout, out_dim)

    def prob_mask_like(self, shape, prob, device):
        if prob == 1:
            return torch.ones(shape, device = device, dtype = torch.bool)
        elif prob == 0:
            return torch.zeros(shape, device = device, dtype = torch.bool)
        else:
            return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

    def forward(self, x, cond, time, context_mask=None, cond_drop_prob=0, cond_scale = 1., rescaled_phi = 0., sampling=False):
        if sampling:
            logits =  self._forward(x, cond, time, context_mask=None, cond_drop_prob=0.)
            if cond_scale == 1:
                return logits
            null_logits =  self._forward(x, cond, time, context_mask=None, cond_drop_prob=1.)
            scaled_logits = null_logits + (logits - null_logits) * cond_scale

            if rescaled_phi == 0.:
                return scaled_logits

            std_fn = partial(torch.std, dim = tuple(range(1, scaled_logits.ndim)), keepdim = True)
            rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

            return rescaled_logits * rescaled_phi + scaled_logits * (1. - rescaled_phi)
        else:
            return self._forward(x, cond, time, context_mask, cond_drop_prob)

    def _forward(self, x, cond, time, context_mask=None, cond_drop_prob=None):
        # context_mask dummy var
        batch_size = x.shape[0]
        # Mapping
        time_emb = self.map_time(time)
        cond_emb = self.map_cond(cond)
        #emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # why swap emb (sin/cos)?
        time_emb = nn.functional.silu(self.map_time_layer(time_emb))
        cond_emb = nn.functional.silu(self.map_cond_layer(cond_emb.reshape(cond_emb.shape[0], -1)))
        #emb = nn.functional.silu(self.map_layer1(emb))
        if cond_drop_prob == None:
            cond_drop_prob = self.cond_drop_prob
        if cond_drop_prob > 0:
            keep_mask = self.prob_mask_like((batch_size,), 1 - cond_drop_prob, device = x.device)
            null_cond_emb = repeat(self.null_emb, 'd -> b d', b = batch_size) 

            cond_emb = torch.where(
                rearrange(keep_mask, 'b -> b 1'),
                cond_emb,
                null_cond_emb
            )
        x = self.first_layer(x)
        for block in self.blocks:
            x = block(x, time_emb, cond_emb)
        x = self.final_layer(nn.functional.silu(x))
        return x

class CtrlResNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, cond_size,
                model_dim      = 128,      # dim multiplier.
                dim_mult        = [1,1,1,1],# dim multiplier for each resblock layer.
                dim_mult_emb    = 4,
                num_blocks          = 4,        # Number of resblocks(mid) per level.
                dropout             = 0.,      # Dropout rate.
                emb_type            = "sinusoidal",# Timestep embedding type
                dim_mult_time  = 1,        # Time embedding size
                dim_mult_cond   = 1,        # Conditional embedding size
                adaptive_scale  = True,     # Feature-wise transformations, FiLM
                skip_scale      = 1.0,      # Skip connection scaling
                affine          = False    # Affine normalization for MLP
                ):

        super().__init__()

        emb_dim = model_dim * dim_mult_emb
        time_dim = model_dim * dim_mult_time
        cond_dim = model_dim * dim_mult_cond
        block_kwargs = dict(dropout = dropout, skip_scale=skip_scale, adaptive_scale=adaptive_scale, affine=affine)

        self.cond_size = cond_size

        self.map_time = PositionalEmbedding(size=time_dim, type=emb_type)
        self.map_cond = PositionalEmbedding(size=cond_dim, type=emb_type)
        self.map_time_layer = nn.Linear(time_dim, emb_dim)
        self.map_cond_layer = nn.Linear(cond_dim*cond_size, emb_dim) # Can also be used for the controlNet (replace Vec2Img)

        self.first_layer = nn.Linear(in_dim, model_dim)
        self.blocks = nn.ModuleList()
        cout = model_dim
        self.numblock = num_blocks

        self.ctrlBlocks = nn.ModuleList()
        self.ctrl_cond_encoder = CondResNetBlock(emb_dim, emb_dim, emb_dim, emb_dim, **block_kwargs)

        for level, mult in enumerate(dim_mult):
            for _ in range(num_blocks):
                cin = cout
                cout = model_dim * mult
                self.blocks.append(ResNetBlock(cin, cout, emb_dim, **block_kwargs))
                '''
                self.ctrlBlocks.append(zero_module(nn.Conv1d(conv_dim, conv_dim, 1)))
                self.ctrlOutBlocks.append(nn.Sequential(zero_module(nn.Conv1d(conv_dim, 1, 1)),
                                                        nn.Flatten(),
                                                        nn.SiLU()))
                '''
            self.ctrlBlocks.append(zero_module(CondResNetBlock(cin, cout, emb_dim, emb_dim, **block_kwargs)))
        self.final_layer = nn.Linear(cout, out_dim)

    def forward(self, x, cond, time, context_mask=None, controlnet=False):
        # Mapping
        time_emb = self.map_time(time)
        cond_emb = self.map_cond(cond)
        #emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # why swap emb (sin/cos)?
        time_emb = nn.functional.silu(self.map_time_layer(time_emb))
        cond_emb = nn.functional.silu(self.map_cond_layer(cond_emb.reshape(cond_emb.shape[0], -1)))
        #emb = nn.functional.silu(self.map_layer1(emb))

        x = self.first_layer(x)
        if controlnet:
            ctrl_cond = self.ctrl_cond_encoder(x, time_emb, cond_emb)
            for level, block in enumerate(self.blocks):
                x = block(x, time_emb)
                if level % self.numblock == 0 and level !=0:
                    ctrl_cond = self.ctrlBlocks[int(level/self.numblock)](ctrl_cond, time_emb, cond_emb)
                    x += ctrl_cond
        else:
            for block in self.blocks:
                x = block(x, time_emb)
        x = self.final_layer(nn.functional.silu(x))
        return x

class CtrlCondResNet(torch.nn.Module):
    def __init__(self, in_dim, out_dim, cond_size,
                model_dim      = 128,      # dim multiplier.
                dim_mult        = [1,1,1,1],# dim multiplier for each resblock layer.
                dim_mult_emb    = 4,
                num_blocks          = 4,        # Number of resblocks(mid) per level.
                dropout             = 0.,      # Dropout rate.
                emb_type            = "sinusoidal",# Timestep embedding type
                dim_mult_time  = 1,        # Time embedding size
                dim_mult_cond   = 1,        # Conditional embedding size
                adaptive_scale  = True,     # Feature-wise transformations, FiLM
                skip_scale      = 1.0,      # Skip connection scaling
                affine          = False    # Affine normalization for MLP
                ):

        super().__init__()

        emb_dim = model_dim * dim_mult_emb
        time_dim = model_dim * dim_mult_time
        cond_dim = model_dim * dim_mult_cond
        block_kwargs = dict(dropout = dropout, skip_scale=skip_scale, adaptive_scale=adaptive_scale, affine=affine)

        self.cond_size = cond_size

        self.map_time = PositionalEmbedding(size=time_dim, type=emb_type)
        self.map_cond = PositionalEmbedding(size=cond_dim, type=emb_type)
        self.map_time_layer = nn.Linear(time_dim, emb_dim)
        self.map_cond_layer = nn.Linear(cond_dim*cond_size, emb_dim) # Can also be used for the controlNet (replace Vec2Img)

        self.first_layer = nn.Linear(in_dim, model_dim)
        self.blocks = nn.ModuleList()
        cout = model_dim
        self.numblock = num_blocks

        self.ctrlBlocks = nn.ModuleList()
        self.ctrl_cond_encoder = CondResNetBlock(emb_dim, emb_dim, emb_dim, emb_dim, **block_kwargs)

        for level, mult in enumerate(dim_mult):
            for _ in range(num_blocks):
                cin = cout
                cout = model_dim * mult
                self.blocks.append(CondResNetBlock(cin, cout, emb_dim, emb_dim, **block_kwargs))
            self.ctrlBlocks.append(zero_module(CondResNetBlock(cin, cout, emb_dim, emb_dim, **block_kwargs)))
        self.final_layer = nn.Linear(cout, out_dim)

    def forward(self, x, cond, time, context_mask=None, controlnet=False):
        # Mapping
        time_emb = self.map_time(time)
        cond_emb = self.map_cond(cond)
        #emb = emb.reshape(emb.shape[0], 2, -1).flip(1).reshape(*emb.shape) # why swap emb (sin/cos)?
        time_emb = nn.functional.silu(self.map_time_layer(time_emb))
        cond_emb = nn.functional.silu(self.map_cond_layer(cond_emb.reshape(cond_emb.shape[0], -1)))
        #emb = nn.functional.silu(self.map_layer1(emb))

        x = self.first_layer(x)
        if controlnet:
            ctrl_cond = self.ctrl_cond_encoder(x, time_emb, cond_emb)
            for level, block in enumerate(self.blocks):
                x = block(x, time_emb, cond_emb)
                if level % self.numblock == 0 and level !=0:
                    ctrl_cond = self.ctrlBlocks[int(level/self.numblock)](ctrl_cond, time_emb, cond_emb)
                    x += ctrl_cond
        else:
            for block in self.blocks:
                x = block(x, time_emb, cond_emb)
        x = self.final_layer(nn.functional.silu(x))
        return x