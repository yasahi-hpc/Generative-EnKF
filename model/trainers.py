from .denoising_diffusion_trainer import DenoisingDiffusionTrainer
from .physics_informed_diffusion_trainer import PhysicsInformedDiffusionTrainer

def get_trainer(name):
    TRAINERS = {
        'Denoising_Diffusion': DenoisingDiffusionTrainer,
        'PhysicsInformedDiffusion': PhysicsInformedDiffusionTrainer,
    }

    for n, t in TRAINERS.items():
        if n.lower() == name.lower():
            return t

    raise ValueError(f'trainer {name} is not defined')
