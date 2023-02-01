# FLAD - Few-shot Learning with Auxiliary Data




## Getting Started:

<details><summary>Environment Setup</summary>
  
First, install python requirements from `requirements.txt` using your favorite virtual environment manager and the appropriate cuda version.

For example, with conda use:
  
```bash
CUDA_VERSION="113"
conda create -n flad_env python==3.8
conda activate flad_env
pip install -r requirements.txt -f https://download.pytorch.org/whl/cu${CUDA_VERSION}/torch_stable.html
```
  
</details>

## To run baselines:
<details><summary>Target-only baseline</summary>
        
```bash
GPU=0
bash target_only_finetuning.sh $GPU
```
</details>

<details><summary>Explore-only baseline</summary>
  
```bash
GPU=0
TARGET_DATASET='copa'
bash all_mixed.sh $GPU $TARGET_DATASET
```
  </details>
  
<details><summary>Exploit-only baseline</summary>
    
```bash
GPU=0
TARGET_DATASET='copa'
bash all_exploit.sh $GPU $TARGET_DATASET
```
    
Note: The exploit-only baseline requires computing auxiliary dataset gradients prior to training.

This process will take a few hours, depending on your hardware.
 </details>

## To run our FLAD methods:
 
<details><summary>EXP3-FLAD</summary>
  
```bash
GPU=0
TARGET_DATASET='copa'
bash all_exp3.sh $GPU $TARGET_DATASET
```
</details>

<details><summary>UCB1-FLAD</summary>
  
```bash
GPU=0
TARGET_DATASET='copa'
bash all_ucb1.sh $GPU $TARGET_DATASET
```

### UCB1-specific info
If you wish to use UCB1, you will need to pre-compute gradient alignments. The trainer class in `trainer.py` will compute and cache the values during training, but if you are planning on running many experiments on a large set of auxiliary datasets, such as P3, you may wish to pre-compute gradients and alignments prior to training with our premade script. The script will compute gradients for a model on each auxiliary dataset and cache them for future use, then compute the alignment with respect to a specific target dataset.

  </details>
  
  
Note: If you plan to run many experiments, it can be useful to pre-compute gradients and gradient alignments.

<details><summary>To pre-compute gradient alignments</summary>
  
You can pre-compute gradients and alignments with:
  
```bash
TARGET_DATASET='copa'
AUXILIARY_DATASET='P3'
python3 src/multirun_create_weight_inits.py --target_dataset $TARGET_DATASET --auxiliary_dataset $AUXILIARY_DATASET
```

NOTE: This script will by default pre-compute gradients with base- and XL-sized T5 models, and T0-3B. To change this, edit the variable `MODELS` found on lines 83-88 of `src/multirun_create_weight_inits.py`.
  
  </details>

### Attribution
The data loading and formatting code is based on [T-Few](https://github.com/r-three/t-few).
