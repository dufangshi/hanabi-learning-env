# Hanabi Learning Env Helper

1. (Linux only) open `environment.yaml`, uncomment `gcc_linux-64` and `gxx_linux-64`.
2. Create the conda environment: `conda env create -f environment.yaml`.
3. Activate it: `conda activate hanabi-env` (or the name you chose).
4. Run `python main.py`; dependencies and the C++ extension will be built automatically on first install.


### Continue training from checkpoint:
```
python MAPPO/runner.py --model_dir results/Hanabi/Hanabi-Full/mappo/check/run17/models --hanabi_mode full
```

### Evaluation and Test
```
python evaluate_model.py --model_dir results/Hanabi/Hanabi-Full/mappo/check/run17/models --iteration latest --hanabi_name Hanabi-Full --num_agents 2 --num_episodes 1000 --num_envs 1 --save_log  --collect_statistics  
```



