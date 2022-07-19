# nnfromscratch

## Setup
- Environments
```
# >= python 3.9
pip install -r requirements.txt
```

- Experiments
    - Add experiment in `config/exp.toml` by adding `[[experiment]]` and its related parameters.
    - `enable` column set to true

## Execution
```
python3 main.py
```