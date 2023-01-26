# FLAD
Few-shot Learning with Auxiliary Data

## Quickstart guide:

First, install python requirements from requirements.txt

To run target-only baseline:
```bash
GPU=0
bash target_only_finetuning.sh $GPU
```

To save on dataloading time for FLAD methods, we recommend to use our multirun scripts which will only load data once for all hyperparameter options.

To run explore-only baseline:
```bash
GPU=0
TARGET_DATASET='copa'
bash all_mixed.sh $GPU $TARGET_DATASET
```

To run exploit-only baseline:
```bash
GPU=0
TARGET_DATASET='copa'
bash all_exploit.sh $GPU $TARGET_DATASET
```

To run EXP3-FLAD:
```bash
GPU=0
TARGET_DATASET='copa'
bash all_exp3.sh $GPU $TARGET_DATASET
```

To run UCB1-FLAD:
```bash
GPU=0
TARGET_DATASET='copa'
bash all_ucb1.sh $GPU $TARGET_DATASET
```
