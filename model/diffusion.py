import math
import pathlib
import torch
import torch.nn.functional as F
import numpy as np
import xarray as xr
from functools import partial
from collections import namedtuple
from torch import nn, einsum
from einops import reduce
from .utils import default, extract, identity, exists
from .schedule import get_schedule

ModelPrediction =  namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        seq_length,
        timesteps = 1000,
        sampling_timesteps = None,
        loss_type = 'l1',
        objective = 'pred_noise',
        beta_schedule = 'cosine_beta',
        p2_loss_weight_gamma = 0.,
        p2_loss_weight_k = 1,
        ddim_sampling_eta = 0.,
        cond_scale = 3.,
        noise_level = 0.1,
        obs_interval = 1,
        chunk_size = 1,
        save_intermediate_sampling_imgs = False,
        diag_steps = 100,
        diag_chunk = 100,
        result_dir = None,
        norm_dict = None,
    ):
        super().__init__()
        self.model = model
        self.channels = self.model.channels
        self.self_condition = self.model.self_condition

        self.seq_length = seq_length

        self.objective = objective
        self.cond_scale = cond_scale
        self.noise_level = noise_level
        self.obs_interval = obs_interval
        self.chunk_size = chunk_size
        self.save_intermediate_sampling_imgs = save_intermediate_sampling_imgs
        self._diag_steps = diag_steps
        self._diag_chunk = diag_chunk
        self._diffusion_dict = {'ddim_sampling_eta': ddim_sampling_eta,
                                'cond_scale': cond_scale,
                                'noise_level': noise_level,
                                'obs_interval': obs_interval,
                                'chunk_size': chunk_size,
                                'diag_steps': diag_steps,
                                'diag_chunk': diag_chunk,
                               }
        if type(norm_dict) is dict:
            self._diffusion_dict = {**self._diffusion_dict, **norm_dict}

        self._result_dir = result_dir
        if self._result_dir is not None:
            self._result_dir = pathlib.Path(self._result_dir)
            if not self._result_dir.exists():
                self._result_dir.mkdir(parents=True)
        else:
            self.save_intermediate_sampling_imgs = False

        self._it = 0
        self._diag_it = 0
        self._imgs_all = []
        self._t_all = []
        self._labels = None

        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        # define schedule
        betas = get_schedule(beta_schedule)(timesteps=timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value = 1.)
        #alphas_cumprod_next = F.pad(alphas_cumprod[1:], (0, 1), value = 0.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        # helper function to register buffer from float64 to float32
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        #register_buffer('alphas_cumprod_next', alphas_cumprod_next)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min =1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        # calculate p2 reweighting
        register_buffer('p2_loss_weight', (p2_loss_weight_k + alphas_cumprod / (1 - alphas_cumprod)) ** -p2_loss_weight_gamma)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond = None, clip_x_start = False, labels = None, rhs = None):
        model_output = self.model.forward_with_cond_scale(x, t, x_self_cond, labels, rhs, cond_scale=self.cond_scale)
        maybe_clip = partial(torch.clamp, min = -1., max = 1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self, x, t, x_self_cond = None, clip_denoised = True, labels = None, rhs = None):
        preds = self.model_predictions(x, t, x_self_cond=x_self_cond, labels=labels, rhs=rhs)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start = x_start, x_t = x, t = t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x, t: int, x_self_cond = None, clip_denoised = True, labels = None, rhs = None):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0],), t, device = x.device, dtype = torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond, clip_denoised=clip_denoised, labels=labels, rhs=rhs)
        noise = torch.randn_like(x) if t > 0 else 0. # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def _save_intermediate_samples(self, name, img, t, labels=None):
        to_numpy = lambda var: var.cpu().numpy()

        img = img.detach().squeeze(axis=-2)
        img = to_numpy(img)
        if labels is not None:
            # observations are the same for all the batches just use first batch
            labels = labels.detach()
            labels = labels[0]
            self._labels = np.squeeze( to_numpy(labels) )

        if self._it % self._diag_steps == 0:
            self._t_all.append(t)
            self._imgs_all.append(img)

            if len(self._t_all) == self._diag_chunk:
                self._to_netcdf(name)

                self._imgs_all = []
                self._t_all = []
                self._labels = None

    def _to_netcdf(self, name):
        img_all = np.asarray(self._imgs_all)
        data_vars = {}
        data_vars['img'] = (['time', 'batch', 'x'], img_all)
        data_vars['time_sampling'] = (['time'], np.asarray(self._t_all))
        if self._labels is not None:
            data_vars['obs'] = (['x'], self._labels)

        _, b, nx  = img_all.shape

        coords = {'time': np.arange(len(self._t_all)),
                  'batch': np.arange(b),
                  'x': np.arange(nx),}
        filenames = list(self._result_dir.glob(f'{name}_sample_it{self._it:05d}_*.nc'))
        diag_it = len(filenames)
        filename = self._result_dir / f'{name}_sample_it{self._it:05d}_{diag_it:03d}.nc'
        ds = xr.Dataset(data_vars=data_vars, coords=coords, attrs=self._diffusion_dict)
        ds.to_netcdf(filename, engine='netcdf4')

    def p_sample_loop(self, shape, labels = None, phys_func = None):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None
        dx = None
        name = 'ddpm'
        for t in reversed(range(0, self.num_timesteps)):
            if self.save_intermediate_sampling_imgs:
                self._save_intermediate_samples(name, img, t, labels)

            if exists(phys_func):
                dx = phys_func(img)

            with torch.no_grad():
                self_cond = x_start if self.self_condition else None
                img, x_start = self.p_sample(img, t, x_self_cond=self_cond, labels=labels, rhs=dx)

        if len(self._t_all):
            self._to_netcdf(name)
            self._imgs_all = []
            self._t_all = []
            self._labels = None

        return img

    def ddim_sample(self, shape, noise = None, clip_denoised = True, labels = None, phys_func = None):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)   # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:])) # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = default(noise, lambda: torch.randn(shape, device = device))

        dx = None
        name = 'ddim'
        for time, time_next in time_pairs:
            if self.save_intermediate_sampling_imgs:
                self._save_intermediate_samples(name, img, time, labels)

            if exists(phys_func):
                dx = phys_func(img)

            with torch.no_grad():
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
                self_cond = x_start if self.self_condition else None
                pred_noise, x_start, *_ = self.model_predictions(img, time_cond, x_self_cond=self_cond, clip_x_start=clip_denoised, labels=labels, rhs=dx)

                if time_next < 0:
                    img = x_start
                    continue

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()

                noise = torch.randn_like(img)

                img = x_start * alpha_next.sqrt() + \
                      c * pred_noise + \
                      sigma * noise

        if len(self._t_all):
            self._to_netcdf(name)
            self._imgs_all = []
            self._t_all = []
            self._labels = None

        return img

    def ddim_reverse_sample(self, x, clip_denoised = True, labels = None, phys_func = None):
        """
        Based on the following implementation
        https://github.com/suxuann/ddib/blob/main/guided_diffusion/gaussian_diffusion.py#L670-L741
        """
        batch, device, total_timesteps, sampling_timesteps, eta, objective = x.shape[0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.arange(end=sampling_timesteps) # [0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(times.int().tolist())
        times_next = times[1:] + [times[-1]]
        time_pairs = list(zip(times, times_next))

        x_t = x.clone()
        dx = None
        name = 'ddim_reverse'
        for time, time_next in time_pairs:
            if self.save_intermediate_sampling_imgs:
                self._save_intermediate_samples(name, x_t, time, labels)

            if exists(phys_func):
                dx = phys_func(x_t)

            with torch.no_grad():
                time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
                self_cond = x if self.self_condition else None
                pred_noise, *_ = self.model_predictions(x_t, time_cond, x_self_cond=self_cond, clip_x_start=clip_denoised, labels=labels, rhs=dx)

                alpha = self.alphas_cumprod[time]
                alpha_next = self.alphas_cumprod[time_next]

                c = (1 - alpha_next).sqrt()
                x0_t = (x_t - pred_noise * (1-alpha).sqrt()) / alpha.sqrt()

                x_t = x0_t * alpha_next.sqrt() + c * pred_noise

        if len(self._t_all):
            self._to_netcdf(name)
            self._imgs_all = []
            self._t_all = []
            self._labels = None

        return x_t

        #for time in times:
        #    time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
        #    self_cond = x if self.self_condition else None
        #    pred_noise, *_ = self.model_predictions(x_t, time_cond, x_self_cond=self_cond, clip_x_start=clip_denoised, labels=labels)

        #    eps = ( extract(self.sqrt_recip_alphas_cumprod, time_cond, x.shape) * x_t - pred_noise ) \
        #        / extract(self.sqrt_recipm1_alphas_cumprod, time_cond, x.shape)

        #    alpha_next = self.alphas_cumprod_next[time]
        #    c = (1 - alpha_next).sqrt()
        #    x_t = pred_noise * alpha_next.sqrt() + eps * c

        #for time, time_next in time_pairs:
        #    time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
        #    self_cond = x if self.self_condition else None
        #    pred_noise, *_ = self.model_predictions(x_t, time_cond, x_self_cond=self_cond, clip_x_start=clip_denoised, labels=labels)

        #    alpha = self.alphas_cumprod[time]
        #    alpha_next = self.alphas_cumprod[time_next]

        #    c = (1 - alpha_next).sqrt()
        #    x0_t = (x_t - pred_noise * (1-alpha).sqrt()) / alpha.sqrt()

        #    #x_t = pred_noise * alpha_next.sqrt() + c * 
        #    x_t = x0_t * alpha_next.sqrt() + c * pred_noise

    def sample(self, batch_size = 16, labels = None, phys_func = None):
        seq_length, channels = self.seq_length, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample
        _sample = sample_fn((batch_size, channels, seq_length), labels=labels, phys_func=phys_func)
        self._it += 1

        return _sample

    def style_transfer(self, x, labels = None, phys_func = None):
        assert self.is_ddim_sampling, 'style transfer needs ddim sampling'
        batch_size = x.shape[0]
        seq_length, channels = self.seq_length, self.channels

        x_noise = self.ddim_reverse_sample(x=x, labels=labels, phys_func=phys_func)
        _sample = self.ddim_sample((batch_size, channels, seq_length), noise=x_noise, labels=labels, phys_func=phys_func)
        self._it += 1

        return _sample

    ###@torch.no_grad()
    ###def interpolate(self, x1, x2, t = None, lam = 0.5):
    ###    b, *_, device = *x1.shape, x1.device
    ###    t = default(t, self.num_timesteps - 1)

    ###    assert x1.shape == x2.shape

    ###    t_batched = torch.stack([torch.tensor(t, device = device)] * b)
    ###    xt1, xt2 = map(lambda x: self.q_sample(x, t = t_batched), (x1, x2))

    ###    img = (1 - lam) * xt1 + lam * xt2
    ###    for i in tqdm(reversed(range(0, t)), desc = 'interpolation sample time step', total = t):
    ###        img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
    ###        
    ###    return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, noise = None, labels = None, phys_func = None):
        b, c, n = x_start.shape

        # epsilon ~ N(0, 1)
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample
        x = self.q_sample(x_start = x_start, t = t, noise = noise)

        # if doing self-conditioning, 50% of the time, predict x_start from current set of times
        # and condition with unet with that
        # this technique will slow down training by 25%, but seems to lower FID significantly

        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        dx = None
        if exists(phys_func):
            dx = phys_func(x)

        model_out = self.model(x, t, x_self_cond, labels, dx)

        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction = 'none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        # Loss for observation data

        loss = loss * extract(self.p2_loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        b, c, n, device, seq_length, = *img.shape, img.device, self.seq_length
        assert n == seq_length, f'seq length must be {seq_length}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        return self.p_losses(img, t, *args, **kwargs)
