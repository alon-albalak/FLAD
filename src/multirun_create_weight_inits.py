#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys

import nltk  # Here to have a nice missing dependency error message early on

import evaluate
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version
from datasets import Dataset

from arguments import (
    ModelArguments,
    DataTrainingArguments,
    TargetDatasetArguments,
    FLADTrainingArguments
)
import argparse
from trainer import FLADSeq2SeqTrainer, BatchedFLADTrainer
from data.data_utils import (
    DatasetWithTemplate,
    FLADWeightedIterableDataset,
    FLADWeightedMapDataset,
    get_train_val_datasets,
    get_test_dataset
)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.23.1")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

MAIN_OUTPUT_DIR=os.path.join(os.path.dirname(__file__),"../outputs")

MODELS = [
    # (model_name_or_path, model_name, gradient_checkpointing, per_device_train_batch_size)
    ("google/t5-base-lm-adapt", "T5_LM_base", False, 16),
    ("bigscience/T0_3B", "T0_3B", True, 8),
    ("google/t5-xl-lm-adapt", "T5_LM_3B", True, 8),
]


def log_dataset_info(split, datasets):
    total_examples = 0
    msg = f"Dataset Metadata ({split}):\n"
    if isinstance(datasets, dict):
        for name, dataset in datasets.items():
            total_examples += len(dataset)
            msg += f"| {name} - {len(dataset)} samples "
    else:
        total_examples = len(datasets)
        msg += f"| {datasets.name} - {len(datasets)} samples "

    msg += f"\nTotal samples: {total_examples}"

    logger.info(msg)

def main(
        model,
        tokenizer,
        data_args,
        target_dataset_args,
        training_args,
        train_dataset,
        validation_dataset,
        target_dataset,
    ):

    # Metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        result = metric.compute(predictions=preds, references=labels)
        return result

    # Initialize our Trainer
    if training_args.gradient_directed and training_args.FLAD_strategy == "batched":
        trainer = BatchedFLADTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            train_dataset_dict=train_dataset_dict if training_args.weight_initialization_samples else None,
            eval_dataset=validation_dataset if training_args.do_eval else None,
            target_dataset=target_dataset if training_args.gradient_directed else None,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            callbacks= [EarlyStoppingCallback(early_stopping_patience=training_args.patience)] if training_args.patience else None,
            similarity_beta=training_args.similarity_beta,
            data_args = data_args,
            target_dataset_args = target_dataset_args,
            weight_init_only=True
        )
    else:
        trainer = FLADSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=validation_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            callbacks= [EarlyStoppingCallback(early_stopping_patience=training_args.patience)] if training_args.patience else None
        )

    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)



TEST_SET_SHOTS = {
    "anli-r1": 50,
    "anli-r2": 50,
    "anli-r3": 50,
    "cb": 32,
    "copa": 32,
    "h-swag": 20,
    "rte": 32,
    "story_cloze": 70,
    "wic": 32,
    "winogrande": 50,
    "wsc": 32
}

if __name__ == "__main__":
    # load command line args
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dataset", type=str)
    parser.add_argument("--auxiliary_dataset", type=str, default="P3")
    parser.add_argument("--train_strategy", type=str, default="auxiliary_and_target")
    parser.add_argument("--gradient_directed", type=bool, default=True)
    args = parser.parse_args()

    # data arguments
    if args.auxiliary_dataset == "P3":
        aux_dataset = "P3"
    else:
        aux_dataset = "T0Mixture"
    data_args = DataTrainingArguments(
            auxiliary_dataset=aux_dataset,
            target_dataset=args.target_dataset,
            )

    WEIGHT_INITIALIZATION_SAMPLES=[100,1000]
    SEEDS=(42, 100, 222, 3456, 5876)

    # method arguments
    per_device_eval_batch_size=64
    overwrite_output_dir=True
    predict_with_generate=True
    evaluation_strategy="steps"
    save_total_limit=1
    patience=0
    load_best_model_at_end=True
    metric_for_best_model="accuracy"
    logging_strategy="steps"
    logging_steps=10
    log_samples_per_dataset=True
    bf16=True
    dataloader_num_workers=0
    optim="adafactor"
    lr_scheduler_type="constant_with_warmup"
    FLAD_strategy="batched"
    similarity_strategy="lm_head"
    dataset_similarity_threshold=None
    weighted_batch_sampling=True
    loss_scaling=False

    max_steps=0
    save_eval_steps=0
    eval_delay=1000
    warmup_ratio=0.01
    beta=0.1
    grad_acc=4
    lr=1e-4

    count = 1
    total = len(MODELS) * len(WEIGHT_INITIALIZATION_SAMPLES) * len(SEEDS)
    # ITERATES THROUGH MODELS - SEEDS - WEIGHT_INITIALIZATION_SAMPLES
    for model_name_or_path, model_name, gradient_checkpointing, per_device_train_batch_size in MODELS:
        model_args = ModelArguments(model_name_or_path=model_name_or_path)

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            use_cache=False if gradient_checkpointing else True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            # device_map="auto"
        )

        for seed in SEEDS:
            # target dataset arguments
            target_dataset_args = TargetDatasetArguments(
                    num_shot=TEST_SET_SHOTS[args.target_dataset],
                    few_shot_random_seed=seed,
                    )
            # data loading
            # Set seed before initializing model.
            set_seed(seed)

            train_dataset, validation_dataset, target_dataset, test_dataset = None, None, None, None

            # Get train/validation datasets
            training_datasets = get_train_val_datasets(args, target_dataset_args, data_args)
            train_dataset = training_datasets[0]
            validation_dataset = training_datasets[1]
            log_dataset_info("train",train_dataset)
            log_dataset_info("validation", validation_dataset)
            # check if doing gradient directed updates
            if len(training_datasets) == 3:
                target_dataset = training_datasets[2]
                log_dataset_info("gradient direction", target_dataset)

            # Get Evaluation/Prediction Datasets
            test_dataset = get_test_dataset(target_dataset_args, data_args)
            log_dataset_info("evaluation", test_dataset)

            # Convert datasets to appropriate torch.dataset items

            if isinstance(train_dataset, dict):
                # If training on a mixture of tasks, use weighted mixture dataset
                train_dataset_dict = {name: DatasetWithTemplate(dataset, tokenizer, include_answer_choices=False) for name, dataset in train_dataset.items()}
                # If weights are none, will initialize with uniform weights
                weights = None

                if not args.gradient_directed:
                    train_dataset = FLADWeightedIterableDataset(train_dataset_dict, weights=weights, seed=seed)
                # If calculating per-sample gradients, use Iterable dataset
                elif FLAD_strategy == "mixed":
                    train_dataset = FLADWeightedIterableDataset(train_dataset_dict, weights=weights, seed=seed)
                # If calculating per-batch gradients, use Map dataset
                else:
                    train_dataset = FLADWeightedMapDataset(train_dataset_dict, weights)
                    target_dataset = DatasetWithTemplate(target_dataset, tokenizer, include_answer_choices=False)

            elif isinstance(train_dataset, Dataset):
                train_dataset = DatasetWithTemplate(train_dataset, tokenizer, include_answer_choices=False)
            
            validation_dataset = DatasetWithTemplate(validation_dataset, tokenizer, include_answer_choices=True, add_special_tokens=True)

            if model.config.decoder_start_token_id is None:
                raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

            if (
                hasattr(model.config, "max_position_embeddings")
                and model.config.max_position_embeddings < data_args.max_source_length
            ):
                if model_args.resize_position_embeddings is None:
                    logger.warning(
                        "Increasing the model's number of position embedding vectors from"
                        f" {model.config.max_position_embeddings} to {data_args.max_source_length}."
                    )
                    model.resize_position_embeddings(data_args.max_source_length)
                elif model_args.resize_position_embeddings:
                    model.resize_position_embeddings(data_args.max_source_length)
                else:
                    raise ValueError(
                        f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
                        f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
                        f" `--max_source_length` to {model.config.max_position_embeddings} or to automatically resize the"
                        " model's position encodings by passing `--resize_position_embeddings`."
                    )

            for weight_initialization_samples in WEIGHT_INITIALIZATION_SAMPLES:
                print(f"*** Running {count}/{total} ***")
                print(f"{model_name} with seed {seed} and weight_initialization_samples {weight_initialization_samples}")
                output_dir = f"{MAIN_OUTPUT_DIR}/weight_inits/{model_name}/{aux_dataset}/{args.target_dataset}/{seed}/{weight_initialization_samples}"
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
    

                # training arguments
                training_args = FLADTrainingArguments(
                    do_train=True,
                    do_eval=True,
                    train_strategy=args.train_strategy,
                    per_device_train_batch_size=per_device_train_batch_size,
                    per_device_eval_batch_size=per_device_eval_batch_size,
                    gradient_accumulation_steps=grad_acc,
                    gradient_checkpointing=gradient_checkpointing,
                    output_dir=output_dir,
                    overwrite_output_dir=overwrite_output_dir,
                    predict_with_generate=predict_with_generate,
                    max_steps=max_steps,
                    evaluation_strategy=evaluation_strategy,
                    eval_steps=save_eval_steps,
                    save_total_limit=save_total_limit,
                    save_steps=save_eval_steps,
                    patience=patience,
                    eval_delay=eval_delay,
                    load_best_model_at_end=load_best_model_at_end,
                    metric_for_best_model=metric_for_best_model,
                    logging_strategy=logging_strategy,
                    logging_steps=logging_steps,
                    log_samples_per_dataset=log_samples_per_dataset,
                    bf16=bf16,
                    learning_rate=lr,
                    dataloader_num_workers=dataloader_num_workers,
                    optim=optim,
                    warmup_ratio=warmup_ratio,
                    lr_scheduler_type=lr_scheduler_type,
                    gradient_directed=args.gradient_directed,
                    FLAD_strategy=FLAD_strategy,
                    similarity_strategy=similarity_strategy,
                    similarity_beta=beta,
                    loss_scaling=loss_scaling,
                    weighted_batch_sampling=weighted_batch_sampling,
                    weight_initialization_samples=weight_initialization_samples,
                    dataset_similarity_threshold=dataset_similarity_threshold,
                    tf32=True
                )

                main(
                    model=model,
                    tokenizer=tokenizer,
                    data_args=data_args,
                    target_dataset_args=target_dataset_args,
                    training_args=training_args,
                    train_dataset=train_dataset,
                    validation_dataset=validation_dataset,
                    target_dataset=target_dataset,
                )
                count += 1