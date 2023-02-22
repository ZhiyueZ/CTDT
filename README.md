# CTDT
Continuous-Time Decision Transformer. AISTATS 2023 (link to be updated). 

Our implementation builds on the codebase of [DT](https://github.com/kzl/decision-transformer). We also include the two new survival environments introduced in the paper in `env`. 

## Installation

```
conda env create -f conda_env.yml
```

## Example usage

Generate a simulation dataset with 

```
python sim_data.py
```

Train a CTDT with:

```
python experiment.py --model_type ctdt --train_model
```

#### Logging

By default, log files are generated in the "work_dir" directory. 
