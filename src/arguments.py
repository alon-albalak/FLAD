# Copyright 2020 The HuggingFace Team. All rights reserved.
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

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from transformers.utils import (
    ExplicitEnum,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    logging,
)

from transformers import Seq2SeqTrainingArguments

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()


logger = logging.get_logger(__name__)
log_levels = logging.get_log_levels_dict().copy()
trainer_log_levels = dict(**log_levels, passive=-1)


class TrainStrategy(ExplicitEnum):
    AUX_ONLY = "auxiliary_only"
    AUX_AND_TARGET = "auxiliary_and_target"
    TARGET_ONLY = "target_only"


class RewardModelPartitions(ExplicitEnum):
    ALL_WEIGHTS = "weight"
    ENCODER = "encoder"
    DECODER = "decoder"
    LM_HEAD = "lm_head"

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )
    auxiliary_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the auxiliary datasets to use if training"
        },
    )
    max_samples_per_auxiliary_dataset: Optional[int] = field(
        default=10000,
        metadata={
            "help": "The maximum number of samples to use per auxiliary dataset"
        }
    )
    target_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the target dataset, if used for training, validation, or testing. "
        }
    )
    train_template_idx: Optional[int] = field(
        default=-1,
        metadata={
            "help": "If using a single template, specify here. -1 is default, uses all templates."
        },
    )
    eval_template_idx: Optional[int] = field(
        default=-1,
        metadata={
            "help": "If using a single template, specify here. -1 is default, uses all templates."
        },
    ),
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum number of samples to use for prediction"
        }
    )
    include_T0_eval: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to include the T0 eval set in the auxiliary training data"
        }
    )
    def __post_init__(self):
        if self.auxiliary_dataset is None and self.target_dataset is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


@dataclass
class TargetDatasetArguments:
    """
    Arguments that are specific to our target datasets used for training and eval.
    """
    num_shot: Optional[int] = field(
        default=None,
        metadata={
            "help": "Specifies the number of samples used for few-shot tasks."
        }
    )
    few_shot_random_seed: Optional[int] = field(
        default=None,
        metadata={
            "help":"Random seed to be used for determining few-shot samples"
        }
    )
    change_hswag_templates: Optional[bool] = field(
        default=True
    )
    raft_cross_validation: Optional[bool] = field(
        default=True
    )
    raft_validation_start: Optional[int] = field(
        default=0
    )
    raft_labels_in_input_string: Optional[str] = field(
        default="comma"
    )
    cleaned_answer_choices_b77: Optional[bool] = field(
        default=True
    )
    def __post_init__(self):
        assert((self.num_shot and self.few_shot_random_seed) or \
                (self.num_shot is None and self.few_shot_random_seed is None)), ""

@dataclass
class FLADTrainingArguments(Seq2SeqTrainingArguments):
    train_strategy: Union[TrainStrategy, str] = field(
        default="auxiliary_and_target",
        metadata={"help": "The training strategy to use, determines which data will be trained on. \
            Options are auxiliary_only, auxiliary_and_target, target_only."
        },
    )
    gradient_directed: Optional[bool] = field(
        default=False,
        metadata={
            "help":
            "Option to use gradients to determine auxiliary dataset \
                sampling when using auxiliary_and_target train_strategy."
            }
    )
    FLAD_strategy: Optional[str] = field(
        default="batched",
        metadata={
            "help": "Determines whether batches are single-dataset or mixed. \
                Options are 'batched' or 'mixed'."
        }
    )
    loss_scaling: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Flag for scaling the loss according to gradient similarity"
        }
    )
    weighted_batch_sampling: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Flag for weighting the batch sampling according to gradient similarities."
        }
    )
    weight_initialization_samples: Optional[int] = field(
        default=0,
        metadata={
            "help": "Number of samples from each auxiliary dataset to use when initializing weights. \
                Defaults to uniform weight distribution when 0."
        }
    )
    precomputed_weight_save_dir: Optional[str] = field(
        default=os.path.dirname(__file__),
        metadata={
            "help": "Path to save precomputed weights to."
        }
    )
    precomputed_grad_save_dir: Optional[str] = field(
        default=os.path.dirname(__file__),
        metadata={
            "help": "Path to save precomputed gradients to."
        }
    )
    dataset_similarity_threshold: Optional[float] = field(
        default=None,
        metadata={
            "help": "Similarity threshold (between -1 and 1) under \
                which datasets will no longer be sampled"
        }
    )
    length_norm: Optional[int] = field(
        default=1,
        metadata={
            "help": "Normalize answer choice scores by length."
        }
    )
    patience: Optional[int] = field(
        default=None,
        metadata={
            "help": "Stop training when the metric specified for \
                `metric_for_best_model` worsend for `patience` number of evaluation calls."
        }
    )
    log_samples_per_dataset: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Flag to log the number of samples seen per dataset"
        }
    )
    relative_sampling_from_target: Optional[float] = field(
        default = -1.,
        metadata={
            "help": "Rate at which to sample from target dataset relative to other datasets."
                    " Only used when train_strategy=auxiliary_and_target and gradient_directed=False"
                    " For sampling rate when gradient_directed=True, see target_training_frequency."
        }
    )
    similarity_beta: Optional[float] = field(
        default=1.,
        metadata={
            "help": "If <1 then gradient similarity updates will be an exponential moving average"
        }
    )
    similarity_strategy: Optional[Union[RewardModelPartitions, str]] = field(
      default="weight",
      metadata={
          "help": "Determines which weights to use for similarity calculation"
                  " Options are: weight, encoder, decoder, lm_head"
      }
    )
    target_training_frequency: Optional[int] = field(
        default=1,
        metadata={
            "help": "Frequency of gradient updates to train on the target task. By default train on the target task before every gradient update."
                    " Only used when gradient_directed=True, see relative_sampling_from_target for gradient_directed=False."
        }
    )
    micro_batch_size: Optional[int] = field(
        default=0,
        metadata={
            "help": "Field for micro-batching, which decomposes a single mini-batch into micro-batches. "
            "If 0, no micro-batching is used, model will be trained with full mini-batch size."
        }
    )
    offload_grads: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Flag to move gradients to CPU for computing similarity. "
            "Useful when using full model gradients."
        }
    )
    exp3: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Flag to use Exp3 algorithm for batch weighting."
        }
    )
    ucb1: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Flag to use UCB1 algorithm for batch weighting."
        }
    )

    def __post_init__(self):
        if not any([self.do_train, self.do_eval, self.do_predict]):
            raise ValueError("Must specify --do_train --do_eval OR --do_predict")
        if self.relative_sampling_from_target != -1.:
            if self.relative_sampling_from_target < 0:
                raise ValueError("relative_sampling_from_target must be non-negative")
            if self.train_strategy != "auxiliary_and_target":
                raise ValueError("Relative sampling from target dataset is only compatible \
                    when training with --train_strategy=auxiliary_and_target")
        if self.gradient_directed and self.FLAD_strategy == "batched":
            if not self.loss_scaling and not self.weighted_batch_sampling:
                raise ValueError("If using batched gradient directed FLAD, must use weighted batch sampling")
        return super().__post_init__()