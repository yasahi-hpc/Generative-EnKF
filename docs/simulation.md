# Lorenz96 model with data assimilation

## About
The Lorenz96 model is a defact standard model in data assimilatin studies. The model is defined as follows.  
![Lorenz96](https://latex.codecogs.com/svg.latex?\frac{dx_{i}}{dt}=\left(x_{i+1}-x_{i-2}\right)x_{i-1}-x_{i}+F\left(\forall{i}=1,\ldots,N\right))  
Here is an illustration of 40-variables Lorenz96 simulation result.
![Lorenz96_img](figs/Perturbed.png) 

## Run
To run simulations, several command line arguments are necessary. Following table summarizes the list of commad line arguments for run mode.
The detailed settings are defined in an input file stored in [cases/simulation](../cases/simulation). If the model works, a symbolic link `<case_name>` to `<out_dir>/<case_name>/results` will be created wherein the simulation results are stored. Both `<case_name>` and `<out_dir>` are set in the input file as described in the following. For DA simulations, we need to complete Nature or Perturbed run beforehand, which is used as Ground Truth (Observations are made for these simulation results) for DA simulations. 

| Run mode | Command | Explanation |
| --- | --- | --- |
| Nature run | ```python run.py --filename dns.json``` | Vanilla Lorenz96 simulation |
| Perturbed run | ```python run.py --filename perturbed.json``` | Lorenz96 simulation with perturbation |
| Run with Nudging | ```python run.py  --model_name Nudging --filename nudging.json``` | Lorenz96 simulation with nudging |
| Ensemble Run with EnKF | ```python run.py --model_name EnKF --filename enkf.json``` | Ensemble simulations with EnKF |
| Ensemble Run with LETKF | ```python run.py --model_name LETKF --filename letkf.json``` | Ensemble simulations with LETKF |
| Ensemble Run without DA | ```python run.py --model_name NoDA --filename debug_kf.json``` | Ensemble simulations without DA |
| Run with ensemble Free DA | ```python run.py --model_name EFDA --filename efda.json``` | Simulation with ensemble free DA method |

### Input parameters
For input parameters, there are three categories `settings`, `grids`, and `simulation`.  
For example, an input json file for [LETKF](../cases/simulation/letkf.json) is 

```json
{
    "settings": {
        "out_dir": "/home/g0/a206230/work/letkf",
        "case_name": "LETKF",
        "in_case_name": "Perturbed"
    },
    "grids": {
        "Nx":40
    },
    "simulation": {
        "model": "DNS",
        "F":8,
        "diag_chunk":10,
        "diag_steps":1,
        "dt":0.05,
        "mode":"default",
        "perturbation_amp":0.00001,
        "nbiter":720,
        "u0_factor":1.001,
        "u0_idx":19,
        "n_ens":32,
        "obs_interval":1,
        "kalman_filter":"letkf",
        "n_local":6,
        "beta":1.0,
        "sigma":1.0
    }
}
```

## Use Lorenz96 simulator to construct dataset
We use Lorenz96 simulator to construct the dataset for the deep learning model.
We run ensemble simulations with different initial values and convert the simulation data to the dataset format. 
The detailed settings are defined in an input file stored in [cases/dataset_factory](../cases/dataset_factory). 

```bash
python run.py --model_name DatasetFactory --filename dataset_factory.json
python convert.py --filename dataset_factory.json --start_idx 0  --end_idx 90  --mode train
python convert.py --filename dataset_factory.json --start_idx 90 --end_idx 95  --mode val
python convert.py --filename dataset_factory.json --start_idx 95 --end_idx 100 --mode test
```

Under the dataset dirctory `<case_name>`, the training, valdiation and test data are placed in the following manner.
```
---/
 |--meta.txt
 |--dataset/
 |  |--train/
 |  |    |--shot000000.nc
 |  |    |--shot000001.nc
 |  |    |--...
 |  |    
 |  |--val/
 |  |    |--shot000000.nc
 |  |    |--shot000001.nc
 |  |    |--... 
 |  |    
 |  └──test/
 |       |--shot000000.nc
 |       |--shot000001.nc
 |       |--... 
 |
 └──results/
    |--log.txt
    |--enkf_stats0000.nc
    |--enkf_stats0000.nc
    |--...
    | 
    |--ens_idx000/
    |--ens_idx001/
    |--...
    └──ens_idx099/
```

## Postscript
To visualize, several command line arguments are necessary. Following table summarizes the list of commad line arguments for postscripting.
The detailed settings are defined in an input file stored in [cases/simulation](../cases/simulation). 
If the model works, a symbolic link `<case_name>` to `<out_dir>/<case_name>/imgs` will be created wherein the simulation results are stored. 
Both `<case_name>` and `<out_dir>` are set in the input file as described in the following. 
For LESs, we need to complete DNS run beforehand, which is used as Ground Truth.

| Run mode | Command | Explanation |
| --- | --- | --- |
| Nature run | ```python post.py --filename dns.json``` | Vanilla Lorenz96 simulation |
| Perturbed run | ```python post.py --filename perturbed.json``` | Lorenz96 simulation with perturbation |
| Run with Nudging | ```python post.py  --model_name Nudging --filename nudging.json``` | Lorenz96 simulation with nudging |
| Ensemble Run with EnKF | ```python post.py --model_name EnKF --filename enkf.json``` | Ensemble simulations with EnKF |
| Ensemble Run with LETKF | ```python post.py --model_name LETKF --filename letkf.json``` | Ensemble simulations with LETKF |
| Ensemble Run without DA| ```python post.py --model_name NoDA --filename debug_kf.json``` | Ensemble simulations without DA |
| Run with ensemble Free DA | ```python post.py --model_name EFDA --filename efda.json``` | Simulation with ensemble free DA method |
| Run with RL-DA | ```python post.py -dirname cases/rl_model --model_name SoftActorCritic --filename soft_actor_critic.json``` | Simulation with RL-DA method |

## Reference
```bibtex
@phdthesis{Lorenz96,
  author = {E.N. Lorenz},
  title = {Predictability: a problem partly solved},
  year = {1995},
  journal = {Seminar on Predictability, 4-8 September 1995},
  volume = {1},
  pages = {1-18},
  month = {1995},
  publisher = {ECMWF},
  address = {Shinfield Park, Reading},
  language = {eng},
}
```
