# Hanabi Learning Env Helper

1. (Linux only) open `environment.yaml`, uncomment `gcc_linux-64` and `gxx_linux-64`.
2. Create the conda environment: `conda env create -f environment.yaml`.
3. Activate it: `conda activate hanabi-env` (or the name you chose).
4. Run `python main.py`; dependencies and the C++ extension will be built automatically on first install.
