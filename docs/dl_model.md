# Deep learning model to surrogate data assimilation

## Training
For example, the arguments for `train.py` on SGI8600 is
```bash
python train.py --filename diffusion.json
```

If the code works appropriately, checkpoint and imgs directories are created under `<case_name>`, which include the following files. To check the model behavior, please check the training curve in `<case_name>/imgs/loss_*.png.` For the elapsed time for training, please look at a log file `<case_name>/log_train_*.txt`.

```
---/
 |--meta.txt
 |--log_train_*.txt
 |
 |--checkpoint/
 |  |--checkpoint000.nc
 |  |--model_checkpoint000.pt
 |  |--...
 |
 └──imgs/
    |--loss_*.png 
    |--train/
    |--val/val_u*.png
    └──test/test_u*.png
```

## Input parameters
For input parameters, there are three categories `settings`, `grids`, and `simulation`.  
For example, an input json file for [diffusion model](../cases/model/diffusion_obs2.json) is

```json
{
    "settings": {
        "in_dir": "/home/g0/a206230/work/letkf/Dataset_factory/dataset",
        "case_name": "diffusion_test",
        "out_dir": "/home/g0/a206230/work/letkf/training"
    },
    "simulation": {
        "F": 8,
        "obs_interval": 2,
        "n_local":6
    },
    "nn_model": {
        "lr": 0.001,
        "beta_1": 0.9,
        "beta_2": 0.999,
        "train_num_steps": 100000,
        "n_freq_eval": 1000,
        "n_freq_sample_checkpoint": 1000,
        "batch_size": 16,
        "prob_uncond": 0.1
    }
}
```

The dataset are loaded from `<in_dir>`, which should be constructed by the simulator. 
The training results will be stored under `<out_dir/case_name>`. By default, the latest model state `model_checkpoint*.pt` is used for the diffusion model in Generative EnKF.
