# FLAD - Few-shot Learning with Auxiliary Data

## Quickstart guide:

First, install python requirements from requirements.txt using your favorite virtual environment manager.

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

### UCB1-specific info
If you wish to use UCB1, you will need to pre-compute gradient alignments. The trainer class in `trainer.py` will compute and cache the values during training, but if you are planning on running many experiments on a large set of auxiliary datasets, such as P3, you may wish to pre-compute gradients and alignments prior to training with our premade script. The script will compute gradients for a model on each auxiliary dataset and cache them for future use, then compute the alignment with respect to a specific target dataset.

You can pre-compute gradients and alignments with:
```bash
TARGET_DATASET='copa'
AUXILIARY_DATASET='P3'
python3 src/multirun_create_weight_inits.py --target_dataset $TARGET_DATASET --auxiliary_dataset $AUXILIARY_DATASET
```

NOTE: This script will by default pre-compute gradients with base- and XL-sized T5 models, and T0-3B. To change this, edit the variable ```MODELS``` found on lines 83-88 of `src/multirun_create_weight_inits.py`.


### Attribution
The data loading and formatting code is based on [T-Few](https://github.com/r-three/t-few).
