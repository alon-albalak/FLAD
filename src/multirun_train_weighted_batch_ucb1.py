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

from datasets.utils.logging import set_verbosity
import nltk  # Here to have a nice missing dependency error message early on

import evaluate
import transformers
from filelock import FileLock
import torch
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version
from datasets import Dataset

from arguments import (
    ModelArguments,
    DataTrainingArguments,
    TargetDatasetArguments,
    MTCLTrainingArguments
)
import argparse
from trainer import MTCLSeq2SeqTrainer, BatchedMTCLTrainer, UCB1BatchedMTCLTrainer
from data.data_utils import (
    DatasetWithTemplate,
    MTCLWeightedIterableDataset,
    MTCLWeightedMapDataset,
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
        model_args,
        data_args,
        target_dataset_args,
        training_args,
        train_dataset,
        validation_dataset,
        target_dataset,
        test_dataset,
        logger
    ):

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
        use_cache=False if training_args.gradient_checkpointing else True
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

    # Convert datasets to appropriate torch.dataset items
    if training_args.do_train:

        if isinstance(train_dataset, dict):
            # If training on a mixture of tasks, use weighted mixture dataset
            train_dataset_dict = {name: DatasetWithTemplate(dataset, tokenizer, include_answer_choices=False) for name, dataset in train_dataset.items()}
            # If weights are none, will initialize with uniform weights
            weights = None
            if training_args.relative_sampling_from_target != -1:
                # if we want to sample more/less frequently from target dataset
                uniform_weight = 1/len(train_dataset_dict)
                weights = [uniform_weight \
                            if name != data_args.target_dataset \
                            else uniform_weight*training_args.relative_sampling_from_target \
                        for name in train_dataset_dict]

            if not training_args.gradient_directed:
                train_dataset = MTCLWeightedIterableDataset(train_dataset_dict, weights=weights, seed=training_args.seed)
            # If calculating per-sample gradients, use Iterable dataset
            elif training_args.mtcl_strategy == "samples":
                train_dataset = MTCLWeightedIterableDataset(train_dataset_dict, weights=weights, seed=training_args.seed)
            # If calculating per-batch gradients, use Map dataset
            else:
                train_dataset = MTCLWeightedMapDataset(train_dataset_dict, weights)
                target_dataset = DatasetWithTemplate(target_dataset, tokenizer, include_answer_choices=False)

        elif isinstance(train_dataset, Dataset):
            train_dataset = DatasetWithTemplate(train_dataset, tokenizer, include_answer_choices=False)
        
        validation_dataset = DatasetWithTemplate(validation_dataset, tokenizer, include_answer_choices=True, add_special_tokens=True)
    
    if training_args.do_eval:
        test_dataset = DatasetWithTemplate(test_dataset, tokenizer, include_answer_choices=True, add_special_tokens=True)
    if training_args.do_predict:
        predict_dataset = DatasetWithTemplate(predict_dataset, tokenizer, include_answer_choices=True, add_special_tokens=True)

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


    # Metric
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        result = metric.compute(predictions=preds, references=labels)
        return result

    # Initialize our Trainer
    if training_args.gradient_directed and training_args.mtcl_strategy == "batched":
        trainer = UCB1BatchedMTCLTrainer(
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
            target_dataset_args = target_dataset_args
        )
    else:
        trainer = MTCLSeq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=validation_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
            callbacks= [EarlyStoppingCallback(early_stopping_patience=training_args.patience)] if training_args.patience else None
        )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(test_dataset, max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


    # TODO - ALON: Haven't touched this yet
    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w") as writer:
                    writer.write("\n".join(predictions))
    return results

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
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target_dataset", type=str)
    parser.add_argument("--aux_dataset", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--weight_initialization_samples", type=int, default=0)
    parser.add_argument("--train_strategy", type=str, default="auxiliary_and_target")
    parser.add_argument("--offload_grads", type=bool, default=False)
    parser.add_argument("--gradient_directed", type=bool, default=True)
    parser.add_argument("--base_output_dir", type=str, default=None)#/share/edc/home/alon_albalak/MTCL/outputs
    args = parser.parse_args()

    # model arguments
    patience=10
    max_steps=10000
    if args.model == "google/t5-base-lm-adapt":
        model_name_or_path = "google/t5-base-lm-adapt"
        model_name = "T5_LM_base"
        per_device_train_batch_size=16
        per_device_eval_batch_size=128
        gradient_checkpointing = False
        betas = [0.1]#, 0.25]
        grad_accs = [16, 32]
        lrs = [3e-4, 1e-4]
    elif args.model == "google/t5-xl-lm-adapt":
        model_name_or_path = "google/t5-xl-lm-adapt"
        model_name = "T5_LM_3B"
        per_device_train_batch_size=8
        per_device_eval_batch_size=64
        gradient_checkpointing = True
        betas = [0.1]#, 0.25]
        grad_accs = [16, 32]
        lrs = [1e-4]
    elif args.model == "bigscience/T0_3B":
        model_name_or_path = "bigscience/T0_3B"
        model_name = "T0_3B"
        per_device_train_batch_size=8
        per_device_eval_batch_size=64
        gradient_checkpointing = True
        betas = [0.1]#, 0.25]
        grad_accs = [16, 32]
        lrs = [1e-4]
    else:
        raise ValueError("Model not supported")

    model_args = ModelArguments(model_name_or_path=model_name_or_path)

    # data arguments
    data_args = DataTrainingArguments(
            auxiliary_dataset=args.aux_dataset,
            target_dataset=args.target_dataset,
            )

    # target dataset arguments
    target_dataset_args = TargetDatasetArguments(
            num_shot=TEST_SET_SHOTS[args.target_dataset],
            few_shot_random_seed=args.seed,
            )
    
    loss_scaling=False
    loss_or_sample_name = "weighted_batch_ucb1"
    weighted_batch_sampling=True
    dataset_similarity_threshold = None
    ucb1=True

    # method arguments
    if args.base_output_dir is not None:
        base_output_dir=f"{args.base_output_dir}/{model_name}/{args.aux_dataset}/"+\
        "{}/"+f"{args.seed}/{args.target_dataset}/"+\
            "{}/{}/{}/"+\
            f"{args.weight_initialization_samples}"
    else:    
        base_output_dir=f"{MAIN_OUTPUT_DIR}/{model_name}/{args.aux_dataset}/"+\
            "{}/"+f"{args.seed}/{args.target_dataset}/"+\
                "{}/{}/{}/"+\
                f"{args.weight_initialization_samples}"
    overwrite_output_dir=True
    predict_with_generate=True
    evaluation_strategy="steps"
    save_total_limit=1
    load_best_model_at_end=True
    metric_for_best_model="accuracy"
    logging_strategy="steps"
    logging_steps=10
    log_samples_per_dataset=True
    bf16=True
    dataloader_num_workers=0
    optim="adafactor"
    warmup_ratio=0.01
    lr_scheduler_type="constant_with_warmup"
    gradient_directed=True
    mtcl_strategy="batched"
    similarity_strategy="lm_head"
    

    # data loading
    # Set seed before initializing model.
    set_seed(args.seed)

    # pre-allocate memory
    tmp_tensor = torch.rand([100000,100000], device='cuda')

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

    # remove pre-allocated memory
    del tmp_tensor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    for lr in lrs:
        for beta in betas:
            for grad_acc in grad_accs:
                if grad_acc == 16:
                    eval_steps=100
                    save_steps=100
                    eval_delay=100
                elif grad_acc == 32:
                    eval_steps=50
                    save_steps=50
                    eval_delay=50

                if dataset_similarity_threshold is not None:
                    # add threshold to output dir
                    output_dir = base_output_dir.format(f"{loss_or_sample_name}_with_threshold", beta, grad_acc, lr, args.weight_initialization_samples)
                    output_dir = output_dir+f"/{dataset_similarity_threshold}"
                else:
                    output_dir = base_output_dir.format(loss_or_sample_name, beta, grad_acc, lr, args.weight_initialization_samples)
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)

                result_file = os.path.join(output_dir, "eval_results.json")
                if os.path.exists(result_file):
                    print(f"Skipping {output_dir} because results already exist")
                    continue

                # See all possible arguments in src/transformers/training_args.py
                # or by passing the --help flag to this script.
                # We now keep distinct sets of args, for a cleaner separation of concerns.

                # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TargetDatasetArguments, MTCLTrainingArguments))
                # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
                #     # If we pass only one argument to the script and it's the path to a json file,
                #     # let's parse it to get our arguments.
                #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
                # else:
                #     model_args, data_args, target_dataset_args, training_args = parser.parse_args_into_dataclasses()

                print(f"Outputting to: {output_dir}")
                log_file = os.path.join(output_dir, "log.log")
                err_file = os.path.join(output_dir, "log.err")
                sys.stdout = open(log_file, "w")
                sys.stderr = open(err_file, "w")

                
                # training arguments
                training_args = MTCLTrainingArguments(
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
                    eval_steps=eval_steps,
                    save_total_limit=save_total_limit,
                    save_steps=save_steps,
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
                    gradient_directed=gradient_directed,
                    mtcl_strategy=mtcl_strategy,
                    similarity_strategy=similarity_strategy,
                    similarity_beta=beta,
                    loss_scaling=loss_scaling,
                    weighted_batch_sampling=weighted_batch_sampling,
                    weight_initialization_samples=args.weight_initialization_samples,
                    dataset_similarity_threshold=dataset_similarity_threshold,
                    offload_grads=args.offload_grads,
                    tf32=True,
                    ucb1=ucb1
                )

                handler = logging.StreamHandler(sys.stdout)
                formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
                handler.setFormatter(formatter)
                logger = logging.getLogger("train_loop")
                log_level = training_args.get_process_log_level()
                logger.setLevel(log_level)
                logger.addHandler(handler)
                set_verbosity(log_level)
                transformers.utils.logging.set_verbosity(log_level)
                transformers.utils.logging.enable_default_handler()
                transformers.utils.logging.enable_explicit_format()

                # Log on each process the small summary:
                logger.warning(
                    f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
                    + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
                )
                # TODO - ALON: Log arguments of interest, not just training args
                logger.info(training_args)
                logger.info(data_args)
                logger.info(target_dataset_args)

                # Detecting last checkpoint.
                last_checkpoint = None
                if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
                    last_checkpoint = get_last_checkpoint(training_args.output_dir)
                    if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                        raise ValueError(
                            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                            "Use --overwrite_output_dir to overcome."
                        )
                    elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                        logger.info(
                            f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                            "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                        )
                try:
                    main(
                        model_args,
                        data_args,
                        target_dataset_args,
                        training_args,
                        train_dataset,
                        validation_dataset,
                        target_dataset,
                        test_dataset,
                        logger
                        )

                    suffixes_to_remove = (
                        ".bin",
                        "spiece.model",
                        "tokenizer_config.json",
                        "special_tokens_map.json",
                        "tokenizer.json",
                    )
                    for root, dirs, files in os.walk(output_dir):
                        for file in files:
                            if file.endswith(suffixes_to_remove):
                                os.remove(os.path.join(root, file))

                except Exception as e:
                    logger.error(e)
                    pass