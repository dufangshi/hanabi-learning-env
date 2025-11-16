# Hanabi Learning Env Helper

1. (Linux only) open `environment.yaml`, uncomment `gcc_linux-64` and `gxx_linux-64`.
2. Create the conda environment: `conda env create -f environment.yaml`.
3. Activate it: `conda activate hanabi-env` (or the name you chose).
4. Run `python main.py`; dependencies and the C++ extension will be built automatically on first install.


### Evaluation
```
python evaluate_model.py --model_dir results/.../models --iteration latest --hanabi_name Hanabi-Full --num_agents 2 --num_episodes 500 --num_envs 10 --save_log      
```


python evaluate_model.py --model_dir results/Hanabi/Hanabi-Full/mappo/check/run14/models --iteration latest --hanabi_name Hanabi-Full --num_agents 2 --num_episodes 2000 --num_envs 10 --save_log      

