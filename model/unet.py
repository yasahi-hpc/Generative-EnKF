import torch
import math
from torch import nn
from functools import partial
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from .utils import default, exists
from .blocks import (
    ResnetBlock,
    RandomOrLearnedSinusoidalPosEmb,
    SinusoidalPosEmb,
    SinusoidalPositionalEmbedding,
    RandomOrLearnedSinusoidalPositionalEmbedding,
    Residual,
    PreNorm,
    LinearAttention,
    Attention,
    Downsample,
    Upsample,
)

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

class TransformerBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        d_model = kwargs.get('d_model', 256)
        d_hidden = kwargs.get('d_hidden', 128)
        nhead = kwargs.get('nhead', 8)
        dim_feedforward = kwargs.get('dim_feedforward', 64)
        dropout = kwargs.get('dropout', 0)
        num_layers = kwargs.get('num_layers', 6)
        self.pos_emb = kwargs.get('pos_emb', 'Sinusoidal')
        num_channels = kwargs.get('num_channels')
        if num_channels is None:
            raise ValueError('Argument num_channels must be given for TransformerBlock')
        w = kwargs.get('w', 40)
        seq_len = kwargs.get('seq_len')
        if seq_len is None:
            raise ValueError('Argument seq_len must be given for TransformerBlock')

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)

        # Transpose and apply linear layer
        # 1. (b t c x) -> (b t (c x))
        # 2. (b t (c x)) -> (b t d)
        self.series_embedding = nn.Sequential(
            Rearrange('b t c x -> b t (c x)'),
            nn.Linear(num_channels*w, d_model),
        )

        if self.pos_emb == 'Sinusoidal':
            self.pos_embedding = SinusoidalPositionalEmbedding(d_model)
        else:
            # [TO DO] Use RandomOrLearnedSinusoidalPositionalEmbedding
            self.pos_embedding = SinusoidalPositionalEmbedding(d_model)

        # Define Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
                                                   dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)

        # Define decoder
        # 1. (b t d) -> (b t (c x))
        # 2. (b t (c x)) -> (b (t c) x)
        # 3. (b (t c) x) -> (b c x)
        self.decoder = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_channels*w),
            nn.GELU(),
            Rearrange('b t (c x) -> b c (t x)', c=num_channels, x=w),
            nn.Linear(w*seq_len, w),
        )

    def forward(self, series):
        """
        series (b, c, t, x)
        """

        # Embedding data (b, c, d)
        out = self.series_embedding(series) * self.scale

        # Positional embedding (b, c, d)
        out = self.pos_embedding(out)

        # Transformer on time series data (b, c, d)
        out = self.transformer_encoder(out)

        # Decode (b, c, d)
        out = self.decoder(out)

        return out

class UNet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        obs_condition = False,
        phys_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        obs_interval = 1,
        chunk_size = 1,
        prob_uncond = 0.1,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        self.obs_condition = obs_condition
        self.phys_condition = phys_condition
        self.obs_interval = obs_interval
        self.prob_uncond = prob_uncond
        self.chunk_size = chunk_size

        num_channel_factor = 1
        if self_condition:
            num_channel_factor += 1
        if obs_condition:
            num_channel_factor += 1
            if self.chunk_size >= 2:
                num_channel_factor += 1

        if phys_condition:
            num_channel_factor += 1

        input_channels = channels * num_channel_factor

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv1d(input_channels, init_dim, 7, padding = 3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups = resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.obs_conv = None
        self.seq_emb = None
        if self.obs_condition:
            self.null = nn.Parameter(torch.randn(channels, init_dim))
            self.seq_emb = TransformerBlock(num_channels=channels, w=init_dim, seq_len=self.chunk_size)

        if self.phys_condition:
            self.null_phys = nn.Parameter(torch.randn(channels, init_dim))

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                block_klass(dim_in, dim_in, time_emb_dim = time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv1d(dim_in, dim_out, 3, padding = 1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim = time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else  nn.Conv1d(dim_out, dim_in, 3, padding = 1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward_with_cond_scale(
        self,
        *args,
        cond_scale = 1.,
        **kwargs
    ):
        eps_cond = self.forward(*args, **kwargs)
        if cond_scale == 1:
            return eps_cond

        eps_uncond = self.forward(*args, prob_uncond=1., **kwargs)
        return eps_uncond + (eps_cond - eps_uncond) * cond_scale

    def forward(self, x, time, x_self_cond=None, obs=None, phys=None, prob_uncond=None):
        batch, device = x.shape[0], x.device

        prob_uncond = default(prob_uncond, self.prob_uncond)

        # Time embedding
        t = self.time_mlp(time)

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim = 1)

        seq = None
        keep_mask = prob_mask_like((batch,), 1 - self.prob_uncond, device=device)

        if self.phys_condition and exists(phys):
            if prob_uncond > 0:
                null_phys = repeat(self.null_phys, 'c d -> b c d', b = batch)

                phys = torch.where(
                    rearrange(keep_mask, 'b -> b 1 1'),
                    phys,
                    self.null_phys
                )
                x = torch.cat((phys, x), dim=1)

        if self.obs_condition and exists(obs):
            # obs (n, t, c, w)
            if self.chunk_size == 1:
                obs = obs.squeeze(dim=1) # Remove time dimension

            elif self.chunk_size >= 2:
                seq = obs.clone()
                obs = obs[:,-1]

                emb = self.seq_emb(seq)
                obs = torch.cat((obs, emb), dim=1)

            if prob_uncond > 0:
                #keep_mask = prob_mask_like((batch,), 1 - self.prob_uncond, device=device)
                null = repeat(self.null, 'c d -> b c d', b = batch)

                obs = torch.where(
                    rearrange(keep_mask, 'b -> b 1 1'),
                    obs,
                    self.null
                )
                x = torch.cat((obs, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)
            
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim = 1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim = 1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)
