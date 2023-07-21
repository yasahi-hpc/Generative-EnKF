import torch
import math

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """

    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)

def quadratic_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps) ** 2

def sigmoid_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    betas = torch.linspace(-6, 6, timesteps)

    sigmoid = lambda x: 1/(1+torch.exp(-x))
    return sigmoid(betas) * (beta_end - beta_start) + beta_start

def get_schedule(name):
    SCHEDULES = {
                 'cosine_beta': cosine_beta_schedule,
                 'linear_beta': linear_beta_schedule,
                 'quadratic_beta': quadratic_beta_schedule,
                 'sigmoid_beta': sigmoid_beta_schedule,
                }

    for n, s in SCHEDULES.items():
        if n.lower() == name.lower():
            return s

    raise ValueError(f'schedule {name} is not defined')
