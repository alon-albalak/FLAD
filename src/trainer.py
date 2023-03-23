from typing import Optional, Dict, Union, Any, Tuple, List, NamedTuple, Callable
import math
import sys
import time
import os
import json
from tqdm.auto import tqdm

import numpy as np
import torch
from torch import nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, RandomSampler, BatchSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import Seq2SeqTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.data.data_collator import DataCollator
from transformers.utils import (
    is_datasets_available,
    ModelOutput,
    logging,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
    is_apex_available
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    LengthGroupedSampler,
    DistributedLengthGroupedSampler,
    find_batch_size,
    nested_numpify
)
from transformers.trainer_utils import (
    seed_worker,
    has_length,
    denumpify_detensorize,
    EvalPrediction,
    ShardedDDPOption,
    HPSearchBackend,
    speed_metrics,
    TrainOutput
)
from transformers.trainer_callback import TrainerState, TrainerCallback
from transformers.deepspeed import deepspeed_init
from transformers.integrations import hp_params
from transformers.pytorch_utils import is_torch_less_than_1_11

if is_datasets_available():
    import datasets

from data.data_utils import (
    FLADBatchSampler,
    FLADDataCollator,
    FLADDistributedBatchSampler,
    FLADWeightedBatchSampler
)
from arguments import FLADTrainingArguments, DataTrainingArguments, TargetDatasetArguments

logger = logging.get_logger(__name__)

TRAINER_STATE_NAME = "trainer_state.json"

if is_apex_available():
    from apex import amp

if is_torch_tpu_available(check_device=False):
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl

if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward

class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    string_predictions: Optional[Union[List[str], Tuple[List[str]]]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    score_candidates: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    score_gts: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]
    idxs: Optional[Union[np.ndarray, Tuple[np.ndarray]]]

class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    string_predictions: Optional[Union[List[str], Tuple[List[str]]]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    idxs: Optional[Union[np.ndarray, Tuple[np.ndarray]]]

class FLADSeq2SeqTrainer(Seq2SeqTrainer):
    def compute_loss(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs: Optional[bool] = False
        ) -> Union[Tuple[torch.Tensor, ModelOutput], torch.Tensor]:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """

        outputs = model(
            input_ids = inputs['input_ids'],
            attention_mask = inputs['attention_mask'],
            decoder_input_ids=inputs['decoder_input_ids'],
            labels=inputs['target_ids']    
        )

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        # if labels is not None:
        #     if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        if isinstance(outputs, dict) and "loss" not in outputs:
            raise ValueError(
                "The model did not return a loss from the inputs, only the following keys: "
                f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
            )
        # We don't use .loss here since the model may return tuples instead of ModelOutput.
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None
            if self.args.world_size <= 1:
                return LengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    generator=generator,
                )
            else:
                return DistributedLengthGroupedSampler(
                    self.args.train_batch_size * self.args.gradient_accumulation_steps,
                    dataset=self.train_dataset,
                    num_replicas=self.args.world_size,
                    rank=self.args.process_index,
                    lengths=lengths,
                    model_input_name=model_input_name,
                    seed=seed,
                )

        else:
            if self.args.gradient_directed and self.args.FLAD_strategy == "batched":
                if self.args.world_size <= 1:
                    if self.args.weighted_batch_sampling:
                        return FLADWeightedBatchSampler(
                            self.train_dataset,
                            batch_size=self.args.train_batch_size,
                            generator=generator,
                            drop_last=self.args.dataloader_drop_last
                        )
                    else:
                        return FLADBatchSampler(
                            self.train_dataset,
                            batch_size=self.args.train_batch_size,
                            generator=generator,
                            drop_last=self.args.dataloader_drop_last
                            )
                else:
                    return FLADDistributedBatchSampler(
                        self.train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed
                    )
            else:
                if self.args.world_size <= 1:
                    return RandomSampler(self.train_dataset, generator=generator)
                else:
                    return DistributedSampler(
                        self.train_dataset,
                        num_replicas=self.args.world_size,
                        rank=self.args.process_index,
                        seed=seed,
                    )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        
        # Get custom collate_fn
        data_collator = FLADDataCollator(
            tokenizer=self.tokenizer,
            model=self.model,
            pad_to_multiple_of=8 if self.args.fp16 else None,
            pretrain=True
        )

        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(train_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                train_dataset = IterableDatasetShard(
                    train_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        train_sampler = self._get_train_sampler()

        if isinstance(train_sampler, BatchSampler):
            return DataLoader(
                train_dataset,
                batch_sampler=train_sampler,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        return DataLoader(
            train_dataset,
            batch_size=self._train_batch_size,
            sampler=train_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def get_eval_dataloader(self,
        eval_dataset: Optional[Dataset] = None
        ) -> DataLoader:
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is a [`~datasets.Dataset`], columns not accepted
                by the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        # Get custom collate_fn
        data_collator = FLADDataCollator(
            tokenizer=self.tokenizer,
            model=self.model,
            pad_to_multiple_of=8 if self.args.fp16 else None,
            pretrain=False
        )

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(eval_dataset, description="evaluation")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="evaluation")

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        return self.get_eval_dataloader(test_dataset)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )    

        # inputs = self._prepare_inputs(inputs)

        input_ids, choices_ids, labels = inputs["input_ids"], inputs["answer_choices_ids"], inputs["labels"]

        input_ids = self._prepare_inputs(input_ids)
        choices_ids = self._prepare_inputs(choices_ids)
        labels = self._prepare_inputs(labels)

        bs, num_choices = choices_ids.size()[:2]
        flat_choices_ids = choices_ids.flatten(0, 1)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()  # [bs, max_seq_len]
        with torch.no_grad():
            encoder_hidden_states = model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
        encoder_hidden_states = encoder_hidden_states.unsqueeze(dim=1).repeat(1, num_choices, 1, 1).flatten(0, 1)
        attention_mask = attention_mask.unsqueeze(dim=1).repeat(1, num_choices, 1).flatten(0, 1)
        decoder_input_ids = torch.cat([torch.zeros_like(flat_choices_ids[:, :1]), flat_choices_ids[:, :-1]], dim=1)
        decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
        lm_target = flat_choices_ids - 100 * (flat_choices_ids == self.tokenizer.pad_token_id).long()

        with torch.no_grad():
            with self.compute_loss_context_manager():
                model_output = model(
                    attention_mask=attention_mask,
                    encoder_outputs=[encoder_hidden_states],
                    decoder_input_ids=decoder_input_ids,
                    decoder_attention_mask=decoder_attention_mask,
                )
        choices_scores = (
            torch.nn.functional.cross_entropy(model_output.logits.flatten(0, 1), lm_target.flatten(0, 1), reduction="none")
            .view(bs, num_choices, -1)
            .sum(dim=-1)
        )
        if self.args.length_norm > 0:
            choices_scores = choices_scores / torch.pow(
                (choices_ids != self.tokenizer.pad_token_id).sum(dim=-1), self.args.length_norm
            )
        pred_score, prediction = choices_scores.min(dim=1)

        score_gt = choices_scores[range(bs), labels]
        choices_scores[range(bs), labels] = choices_scores.max(dim=-1)[0]
        score_cand = choices_scores.min(dim=-1)[0]

        # batch_output = {
        #     "prediction": prediction.tolist(),
        #     "label": labels.tolist(),
        #     "idx": inputs["idx"].tolist(),
        #     "log.score_gt": score_gt.tolist(),
        #     "log.score_cand": score_cand.tolist(),
        # }
        # return batch_output

        return prediction, labels, inputs['idx'], score_gt, score_cand

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        if hasattr(dataloader.dataset.dataset, "templates"):
            answer_choices = dataloader.dataset.dataset.templates[0].answer_choices
            # if len(dataloader.dataset.dataset.templates) > 1:
            #     assert all([answer_choices == t.answer_choices for t in dataloader.dataset.dataset.templates[1:]]), "all templates should have the same answer choices"
        else:
            answer_choices = None

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        preds_host = None
        labels_host = None
        idx_host = None
        score_gt_host = None
        score_cand_host = None

        # losses/preds/labels on CPU (final containers)
        all_preds = None
        all_labels = None
        all_idx = None
        all_score_gt = None
        all_score_cand = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            preds, labels, idx, score_gt, score_cand = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            # predictions = self._pad_across_processes(preds)
            # predictions = self._nested_gather(predictions)
            # preds_host = preds if preds_host is None else nested_concat(preds_host, predictions, padding_idx=-100)
            preds_host = preds if preds_host is None else torch.cat((preds_host, preds), dim=0)
            labels_host = labels if labels_host is None else torch.cat((labels_host, labels), dim=0)
            idx_host = idx if idx_host is None else torch.cat((idx_host, idx), dim=0)
            score_gt_host = score_gt if score_gt_host is None else torch.cat((score_gt_host, score_gt), dim=0)
            score_cand_host = score_cand if score_cand_host is None else torch.cat((score_cand_host, score_cand), dim=0)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if preds_host is not None:
                    predictions = nested_numpify(preds_host)
                    all_preds = predictions if all_preds is None else np.concatenate((all_preds, predictions), axis=0)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = labels if all_labels is None else np.concatenate((all_labels, labels), axis=0)
                if idx_host is not None:
                    idxs = nested_numpify(idx_host)
                    all_idx = idxs if all_idx is None else np.concatenate((all_idx, idxs), axis=0)
                if score_gt_host is not None:
                    score_gts = nested_numpify(score_gt_host)
                    all_score_gt = score_gts if all_score_gt is None else np.concatenate((all_score_gt, score_gts), axis=0)
                if score_cand_host is not None:
                    score_cands = nested_numpify(score_cand_host)
                    all_score_cand = score_cands if all_score_cand is None else np.concatenate((all_score_cand, score_cands), axis=0)

                # Set back to None to begin a new accumulation
                preds_host, labels_host, idx_host, score_gt_host, score_cand_host = None, None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if preds_host is not None:
            predictions = nested_numpify(preds_host)
            all_preds = predictions if all_preds is None else np.concatenate((all_preds, predictions), axis=0)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else np.concatenate((all_labels, labels), axis=0)
        if idx_host is not None:
            idxs = nested_numpify(idx_host)
            all_idx = idxs if all_idx is None else np.concatenate((all_idx, idxs), axis=0)
        if score_gt_host is not None:
            score_gts = nested_numpify(score_gt_host)
            all_score_gt = score_gts if all_score_gt is None else np.concatenate((all_score_gt, score_gts), axis=0)
        if score_cand_host is not None:
            score_cands = nested_numpify(score_cand_host)
            all_score_cand = score_cands if all_score_cand is None else np.concatenate((all_score_cand, score_cands), axis=0)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        # Convert predictions to string answer choices
        if answer_choices is not None and all_preds is not None:
            string_predictions = [answer_choices[pred] for pred in all_preds]
        else:
            string_predictions = None


        return EvalLoopOutput(
            predictions=all_preds,
            string_predictions=string_predictions,
            label_ids=all_labels,
            score_candidates=all_score_cand,
            score_gts=all_score_gt,
            metrics=metrics,
            num_samples=num_samples,
            idxs=all_idx,
            )

    def predict(
        self,
        test_dataset: Dataset,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "test",
        **gen_kwargs
    ) -> PredictionOutput:
        """
        Run prediction and returns predictions and potential metrics.

        Depending on the dataset and your use case, your test dataset may contain labels. In that case, this method
        will also return metrics, like in `evaluate()`.

        Args:
            test_dataset (`Dataset`):
                Dataset to run the predictions on. If it is a [`~datasets.Dataset`], columns not accepted by the
                `model.forward()` method are automatically removed. Has to implement the method `__len__`
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.
            metric_key_prefix (`str`, *optional*, defaults to `"eval"`):
                An optional prefix to be used as the metrics key prefix. For example the metrics "bleu" will be named
                "eval_bleu" if the prefix is `"eval"` (default)
            max_length (`int`, *optional*):
                The maximum target length to use when predicting with the generate method.
            num_beams (`int`, *optional*):
                Number of beams for beam search that will be used when predicting with the generate method. 1 means no
                beam search.
            gen_kwargs:
                Additional `generate` specific kwargs.

        <Tip>

        If your predictions or labels have different sequence lengths (for instance because you're doing dynamic
        padding in a token classification task) the predictions will be padded (on the right) to allow for
        concatenation into one array. The padding index is -100.

        </Tip>

        Returns: *NamedTuple* A namedtuple with the following keys:

            - predictions (`np.ndarray`): The predictions on `test_dataset`.
            - label_ids (`np.ndarray`, *optional*): The labels (if the dataset contained some).
            - metrics (`Dict[str, float]`, *optional*): The potential dictionary of metrics (if the dataset contained
              labels).
        """

        gen_kwargs = gen_kwargs.copy()
        if gen_kwargs.get("max_length") is None and gen_kwargs.get("max_new_tokens") is None:
            gen_kwargs["max_length"] = self.args.generation_max_length
        gen_kwargs["num_beams"] = (
            gen_kwargs["num_beams"] if gen_kwargs.get("num_beams") is not None else self.args.generation_num_beams
        )
        self._gen_kwargs = gen_kwargs

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        test_dataloader = self.get_test_dataloader(test_dataset)
        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
            test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
        )
        total_batch_size = self.args.eval_batch_size * self.args.world_size
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return PredictionOutput(predictions=output.predictions, string_predictions=output.string_predictions, label_ids=output.label_ids, metrics=output.metrics, idxs=output.idxs)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        if args.log_samples_per_dataset:
            samples_seen_per_dataset = {}
        else:
            samples_seen_per_dataset = None

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                sample_datasets = inputs.pop("dataset_name")
                if args.log_samples_per_dataset:
                    for dataset_name in sample_datasets:
                        if dataset_name not in samples_seen_per_dataset:
                            samples_seen_per_dataset[dataset_name] = 0
                        samples_seen_per_dataset[dataset_name] += 1

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step = self.training_step(model, inputs)
                else:
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, samples_seen_per_dataset)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, samples_seen_per_dataset)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, samples_seen_per_dataset):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs['step'] = self.state.global_step
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            if samples_seen_per_dataset is not None:
                logs["samples_seen_per_dataset"] = {k: v for k,v in samples_seen_per_dataset.items()}

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)


class BatchedFLADTrainer(FLADSeq2SeqTrainer):
    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module] = None,
        args: FLADTrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        train_dataset_dict: Optional[Dict] = None,
        eval_dataset: Optional[Dataset] = None,
        target_dataset: Optional[Dataset] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Callable[[], PreTrainedModel] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
        similarity_beta: Optional[float] = 1,
        data_args: DataTrainingArguments = None,
        target_dataset_args: TargetDatasetArguments = None,
        
    ):
        super().__init__(
            model,args, data_collator,train_dataset,eval_dataset,
            tokenizer,model_init,compute_metrics,callbacks,optimizers,preprocess_logits_for_metrics
        )
        self.auxiliary_dataset_names = train_dataset.dataset_names
        self.target_dataset = target_dataset
        self.target_name = target_dataset.dataset_name
        self.similarity_beta = similarity_beta
        self.train_dataset_dict = train_dataset_dict
        self.data_args = data_args
        self.target_dataset_args = target_dataset_args
        # Initialize gradients and similarities
        self._initialize_grads_similarities()
        self._vars_to_log = {}
        self._initialize_vars_to_log()

    def _initialize_vars_to_log(self):
        # add variable names and their value type to self._vars_to_log
        self._vars_to_log["similarities"] = "tensor"

    def _training_step_large_model(self, model, inputs, batch_dataset, scale_by_similarities):
        batch_size = inputs["input_ids"].shape[0]
        batch_keys = list(inputs.keys())
        micro_batch_loss = 0
        for i in range(0, batch_size, self.args.micro_batch_size):
            micro_inputs = {k: inputs[k][i:i+self.args.micro_batch_size] for k in batch_keys}

            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, micro_inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, micro_inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # scale loss by similarity to target dataset
            if scale_by_similarities:
                loss = loss*self._similarities[batch_dataset]

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()

            micro_batch_loss += loss

        return micro_batch_loss

    def training_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            batch_dataset: Optional[str] = None,
            scale_by_similarities: Optional[bool] = False,
            return_grads: Optional[bool] = False
        ) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        # gather accumulated gradients
        if return_grads:
            with torch.no_grad():
                prev_grads = {}
                first_param = True
                for name, param in model.named_parameters():
                    if self.args.reward_model_partition in name:
                        if first_param:
                            if param.grad is None or torch.sum(param.grad) == 0:
                                break
                            else:
                                first_param = False
                        prev_grads[name] = param.grad.detach().clone()
                        if self.args.offload_grads:
                            # move gradients to cpu
                            prev_grads[name] = prev_grads[name].cpu()

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.args.micro_batch_size > 0:
            loss = self._training_step_large_model(model, inputs, batch_dataset, scale_by_similarities)

        else:
            if is_sagemaker_mp_enabled():
                loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
                return loss_mb.reduce_mean().detach().to(self.args.device)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if self.args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training

            # scale loss by similarity to target dataset
            if scale_by_similarities:
                loss = loss*self._similarities[batch_dataset]

            if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps

            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()

        # gather gradients
        if return_grads:
            with torch.no_grad():
                grads = []
                for name, param in model.named_parameters():
                    if self.args.reward_model_partition in name:
                        p = param.grad.detach()
                        if self.args.offload_grads:
                            p = p.cpu()
                            
                        if prev_grads:
                            p = p - prev_grads[name]

                        grads.append(p)
                grads = torch.concat([g.flatten() for g in grads])
            return loss.detach(), grads

        return loss.detach()

    def _get_target_sampler(self, target_dataset=None) -> Optional[torch.utils.data.Sampler]:
        target_dataset = target_dataset if target_dataset else self.target_dataset
        
        if target_dataset is None or not has_length(target_dataset):
            return None

        generator = None
        if self.args.world_size <= 1:
            generator = torch.Generator()
            # for backwards compatibility, we generate a seed here (which is sampled from a generator seeded with
            # `args.seed`) if data_seed isn't provided.
            # Further on in this method, we default to `args.seed` instead.
            if self.args.data_seed is None:
                seed = int(torch.empty((), dtype=torch.int64).random_().item())
            else:
                seed = self.args.data_seed
            generator.manual_seed(seed)

        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed

        # Build the sampler.
        if self.args.world_size <= 1:
            return RandomSampler(target_dataset, generator=generator)
        else:
            return DistributedSampler(
                target_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=seed,
            )

    def get_target_dataloader(self, target_dataset = None) -> DataLoader:
        target_dataset = target_dataset if target_dataset else self.target_dataset
        
        if target_dataset is None:
            raise ValueError("Trainer: Gradient directed training requires a target_dataset.")
        
        # Get custom collate_fn
        data_collator = FLADDataCollator(
            tokenizer=self.tokenizer,
            model=self.model,
            pad_to_multiple_of=8 if self.args.fp16 else None,
            pretrain=True
        )

        if is_datasets_available() and isinstance(target_dataset, datasets.Dataset):
            target_dataset = self._remove_unused_columns(target_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        if isinstance(target_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                target_dataset = IterableDatasetShard(
                    target_dataset,
                    batch_size=self._train_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )

            return DataLoader(
                target_dataset,
                batch_size=self.args.per_device_train_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        target_sampler = self._get_target_sampler(target_dataset)

        return DataLoader(
            target_dataset,
            batch_size=self._train_batch_size,
            sampler=target_sampler,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            worker_init_fn=seed_worker,
        )

    def _calculate_grad(self, model, dataloader):
        # iterate through dataloader and accumulate gradients over all samples
        # return the gradient
        grad = None
        for step, inputs in enumerate(dataloader):
            if self.args.local_rank != -1:
                with model.no_sync():
                    loss = self.training_step(model, inputs, self.target_name)
            else:
                loss = self.training_step(model, inputs, self.target_name)


        with torch.no_grad():
            grad = []
            for name, param in model.named_parameters():
                # print(name)
                if self.args.reward_model_partition in name:
                    # if "weight" in name:
                    grad.append(param.grad.detach())
            grad = torch.concat([g.flatten() for g in grad])
        model.zero_grad()
        return grad

    def _update_grad(self, task_name, grad):
        if self._gradients[task_name] is None:
            self._gradients[task_name] = grad
        else:
            self._gradients[task_name] += grad

    def _clear_grads(self):
        self._gradients = {name: None for name in self.auxiliary_dataset_names}

    def _initialize_grads_similarities(self):
        # Initialize gradients and gradient similarities
        self._gradients = {name: None for name in self.auxiliary_dataset_names}
        self._similarities = {name: torch.tensor(1, dtype=torch.float) for name in self.auxiliary_dataset_names}

    def _calculate_similarity(self, target_grad, auxiliary_grad):
        if self.args.reward_function == "cosine":
            # calculate cosine similarity between grads and target
            return torch.nn.functional.cosine_similarity(target_grad, auxiliary_grad, dim=0)            
        elif self.args.reward_function == "magnitude":
            # calculate magnitude similarity between grads and target
            target_norm = torch.linalg.norm(target_grad)
            auxiliary_norm = torch.linalg.norm(auxiliary_grad)
            return (2*target_norm*auxiliary_norm + 1e-8) / \
                (target_norm**2 + auxiliary_norm**2 + 1e-8)
        elif self.args.reward_function == "cosine_and_magnitude":
            # calculate cosine similarity between grads and target
            cosine_similarity = torch.nn.functional.cosine_similarity(target_grad, auxiliary_grad, dim=0)
            # normalize cosine_similarity to 0-1 range
            cosine_similarity = (cosine_similarity + 1) / 2
            # calculate magnitude similarity between grads and target
            target_norm = torch.linalg.norm(target_grad)
            auxiliary_norm = torch.linalg.norm(auxiliary_grad)
            magnitude_similarity = (2*target_norm*auxiliary_norm + 1e-8) / \
                (target_norm**2 + auxiliary_norm**2 + 1e-8)
            return cosine_similarity + magnitude_similarity
        else:
            raise ValueError("Invalid similarity strategy.")

    def _update_grad_similarity(self, target_grad):
        # calculate similarity between grads and target
        for dataset_name in self.auxiliary_dataset_names:
            if self._gradients[dataset_name] is not None:
                sim = self._calculate_similarity(target_grad, self._gradients[dataset_name])
                self._similarities[dataset_name] = \
                    (1-self.similarity_beta)*self._similarities[dataset_name] + \
                        self.similarity_beta*sim

                # move similarity to device if needed
                if self.args.offload_grads:
                    self._similarities[dataset_name] = self._similarities[dataset_name].to(self.args.device)

                if (self.args.dataset_similarity_threshold is not None) and \
                    (self._similarities[dataset_name] < self.args.dataset_similarity_threshold):
                    self._similarities[dataset_name] = self._similarities[dataset_name] * 0

    def _update_dataloader_weights(self, dataloader):
        weights = torch.tensor(list(self._similarities.values()))
        dataloader.batch_sampler.update_weights_and_distribution(weights, \
                                    threshold=self.args.dataset_similarity_threshold)

    def _initialize_weights(self, train_dataloader, target_dataloader, model):
        # Initialize weights for each dataset

        # rename models with symbols incompatible with file names
        model_name = model.name_or_path.replace("/","-")

        # save path for weights
        weight_save_path = os.path.join(self.args.precomputed_weight_save_dir,"initial_similarities")
        if not os.path.exists(weight_save_path):
            os.makedirs(weight_save_path, exist_ok=True)
        weight_save_file = os.path.join(weight_save_path,
            f"{self.args.weight_initialization_samples}_{self.data_args.target_dataset}_"
            f"{self.data_args.auxiliary_dataset}_{model_name}_{self.target_dataset_args.few_shot_random_seed}.json"
            )
        if self.args.reward_function == "magnitude":
            weight_save_file = weight_save_file.replace(".json","_magnitude.json")
        elif self.args.reward_function == "cosine_and_magnitude":
            weight_save_file = weight_save_file.replace(".json","_cosine_and_magnitude.json")

        # save path for gradients
        grad_save_path = os.path.join(self.args.precomputed_grad_save_dir,"initial_gradients",
            f"{self.args.weight_initialization_samples}_{self.data_args.auxiliary_dataset}_"
            f"{model_name}"
        )

        if not os.path.exists(grad_save_path):
            os.makedirs(grad_save_path, exist_ok=True)

        logger.info(f"Initializing weights with {self.args.weight_initialization_samples} samples per dataset")

        # load weights if they exist
        if os.path.exists(weight_save_file):
            logger.info(f"Loading weights from {weight_save_file}")
            similarities = json.load(open(weight_save_file))
            for name in self.train_dataset_dict:
                self._similarities[name] = torch.tensor(similarities[name], dtype=torch.float).to(model.device)
        else:
            # calculate similarities
            similarities = {}
            target_grad = self._calculate_grad(model, target_dataloader)
            for name, dataset in tqdm(self.train_dataset_dict.items(), desc="Weight Initialization"):
                save_name = name.replace("/","-")
                grad_save_file = os.path.join(grad_save_path, f"{save_name}.pt")
                if os.path.exists(grad_save_file):
                    grad = torch.load(grad_save_file)
                else:
                    num_samples = min(self.args.weight_initialization_samples, len(dataset))
                    d = torch.utils.data.dataset.Subset(dataset, range(0,num_samples))
                    dataloader = self.get_target_dataloader(d)
                    grad = self._calculate_grad(model, dataloader)
                    torch.save(grad, grad_save_file)

                sim = self._calculate_similarity(target_grad, grad)
                similarities[name] = sim.detach().clone().item()
                logger.info(f"Initial similarity for {name}: {similarities[name]}")
                self._similarities[name] = sim.detach().clone()
            with open(weight_save_file, 'w') as f:
                json.dump(similarities, f, indent=2)

        if self.args.weighted_batch_sampling:
            self._update_dataloader_weights(train_dataloader)

        logger.info(f"Initial similarities: {self._similarities}")

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        target_dataloader = self.get_target_dataloader(self.target_dataset)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        if args.log_samples_per_dataset:
            samples_seen_per_dataset = {}
        else:
            samples_seen_per_dataset = None

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Initialize weights if needed
        if self.args.weight_initialization_samples > 0:
            self._initialize_weights(train_dataloader, target_dataloader, model)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):            
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                sample_datasets = inputs.pop("dataset_name")
                assert(all([s == sample_datasets[0] for s in sample_datasets]))
                batch_dataset = sample_datasets[0]
                if args.log_samples_per_dataset:
                    if batch_dataset not in samples_seen_per_dataset:
                        samples_seen_per_dataset[batch_dataset] = 0
                    samples_seen_per_dataset[batch_dataset] += len(sample_datasets)

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step, grad_step = self.training_step(
                                                    model,
                                                    inputs,
                                                    batch_dataset,
                                                    scale_by_similarities=self.args.loss_scaling,
                                                    return_grads=True
                                                    )
                else:
                    tr_loss_step, grad_step = self.training_step(
                                                model,
                                                inputs,
                                                batch_dataset,
                                                scale_by_similarities=self.args.loss_scaling,
                                                return_grads=True
                                                )

                self._update_grad(batch_dataset, grad_step)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    
                    if (step + 1) % args.target_training_frequency == 0:
                        # Train on full training set before gradient update
                        for step, inputs in enumerate(target_dataloader):
                            tr_loss_step = self.training_step(
                                                        model,
                                                        inputs,
                                                        batch_dataset,
                                                        scale_by_similarities=False,
                                                        return_grads=False
                                                        )
                            tr_loss += tr_loss_step

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, samples_seen_per_dataset)

                    target_grad = self._calculate_grad(model, target_dataloader).detach()
                    if self.args.offload_grads:
                        target_grad = target_grad.cpu()
                        
                    self._update_grad_similarity(target_grad)
                    self._clear_grads()
                    if self.args.weighted_batch_sampling:
                        self._update_dataloader_weights(train_dataloader)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, samples_seen_per_dataset)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        # Check if we've run eval
        if self.state.global_step > args.eval_steps:
            train_loss = self._total_loss_scalar / self.state.global_step
        else:
            # if not, do so manually
            self.manual_eval_and_save(model, trial, ignore_keys_for_eval)
            train_loss = 0.0

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, samples_seen_per_dataset):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs['step'] = self.state.global_step
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            if samples_seen_per_dataset is not None:
                logs["samples_seen_per_dataset"] = {k: v for k,v in samples_seen_per_dataset.items()}
            for var,t in self._vars_to_log.items():
                if t == "scalar":
                    logs[var] = {k:v for k,v in getattr(self, f"_{var}").items()}
                elif t == "tensor":
                    logs[var] = {k:v.item() for k,v in getattr(self, f"_{var}").items()}

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def manual_eval_and_save(self, model, trial, ignore_keys_for_eval):
        if isinstance(self.eval_dataset, dict):
            for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                metrics = self.evaluate(
                    eval_dataset=eval_dataset,
                    ignore_keys=ignore_keys_for_eval,
                    metric_key_prefix=f"eval_{eval_dataset_name}",
                )
        else:
            metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
        self._report_to_hp_search(trial, self.state.global_step, metrics)

        self._save_checkpoint(model, trial, metrics=metrics)
        self.control = self.callback_handler.on_save(self.args, self.state, self.control)

class Exp3BatchedFLADTrainer(BatchedFLADTrainer):
    def _initialize_grads_similarities(self):
        # Initialize gradients and gradient similarities
        self._gradients = {name: None for name in self.auxiliary_dataset_names}
        self._similarities = {name: torch.tensor(1, dtype=torch.float) for name in self.auxiliary_dataset_names}
        self._cumulative_estimated_reward = {name: torch.tensor(0, dtype=torch.float) for name in self.auxiliary_dataset_names}
        self._probabilities = {name: None for name in self.auxiliary_dataset_names}
        self.eps = 1/len(self.auxiliary_dataset_names)
        self.prev_eps = None
        self._update_probabilities()

    def _initialize_vars_to_log(self):
        self._vars_to_log['cumulative_estimated_reward'] = "tensor"
        self._vars_to_log['probabilities'] = "tensor"
        self._vars_to_log['similarities'] = "tensor"

    def _initialize_weights(self, train_dataloader, target_dataloader, model):
        super()._initialize_weights(train_dataloader, target_dataloader, model)
        self._update_probabilities()

    def _update_grad_similarity(self, target_grad):
        # calculate similarity between grads and target
        for dataset_name in self.auxiliary_dataset_names:
            if self._gradients[dataset_name] is not None:
                sim = self._calculate_similarity(target_grad, self._gradients[dataset_name])
                self._similarities[dataset_name] = \
                    (1-self.similarity_beta)*self._similarities[dataset_name] + \
                        self.similarity_beta*sim

                # Exp3 update
                self._cumulative_estimated_reward[dataset_name] = self._cumulative_estimated_reward[dataset_name] + \
                    ((self._similarities[dataset_name])/self._probabilities[dataset_name])

                # move similarity to device if needed
                if self.args.offload_grads:
                    self._similarities[dataset_name] = self._similarities[dataset_name].to(self.args.device)

                if (self.args.dataset_similarity_threshold is not None) and \
                    (self._similarities[dataset_name] < self.args.dataset_similarity_threshold):
                    self._similarities[dataset_name] = self._similarities[dataset_name] * 0
        self._update_probabilities()

    def _update_probabilities(self):
        # calculate epsilons
        self.prev_eps = self.eps
        self.eps = min(1/len(self.auxiliary_dataset_names), np.sqrt(np.log(len(self.auxiliary_dataset_names))/(len(self.auxiliary_dataset_names)*((self.state.global_step*self.args.gradient_accumulation_steps*10)+1))))

        # calculate scaling factor
        tot_estimated_rewards = torch.sum(torch.exp(torch.tensor(list(self._cumulative_estimated_reward.values()))*self.prev_eps))
        scaling_factor = (1-len(self.auxiliary_dataset_names)*self.eps)/tot_estimated_rewards
        
        # update probabilities
        for dataset_name in self.auxiliary_dataset_names:
            self._probabilities[dataset_name] = \
                torch.exp(self.prev_eps*self._cumulative_estimated_reward[dataset_name])*scaling_factor + self.eps

    def _update_dataloader_weights(self, dataloader):
        weights = torch.tensor(list(self._probabilities.values()))
        dataloader.batch_sampler.update_weights_and_distribution(weights, \
                                    threshold=self.args.dataset_similarity_threshold)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, samples_seen_per_dataset):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs['step'] = self.state.global_step
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            if samples_seen_per_dataset is not None:
                logs["samples_seen_per_dataset"] = {k: v for k,v in samples_seen_per_dataset.items()}
            for var,t in self._vars_to_log.items():
                if t == "scalar":
                    logs[var] = {k:v for k,v in getattr(self, f"_{var}").items()}
                elif t == "tensor":
                    logs[var] = {k:v.item() for k,v in getattr(self, f"_{var}").items()}
            # logs['cumulative_estimated_reward'] = {k: v.item() for k,v in self._cumulative_estimated_reward.items()}
            # logs['probabilities'] = {k: v.item() for k,v in self._probabilities.items()}
            # logs['gradient_similarities'] = {k: v.item() for k,v in self._similarities.items()}

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

class UCB1BatchedFLADTrainer(BatchedFLADTrainer):
    def _update_dataloader_weights(self, dataloader, batches_per_dataset=None, step_remainder=0):
        # UCB1 update
        played_rounds = self.state.global_step*self.args.gradient_accumulation_steps+\
            step_remainder+len(self.auxiliary_dataset_names)
        best_action_idx, best_action_value = None, float('-inf')
        for i, dataset_name in enumerate(self.auxiliary_dataset_names):
            if batches_per_dataset is None:
                val = self._similarities[dataset_name] + np.sqrt(2*np.log(played_rounds))
            else:
                val = self._similarities[dataset_name] + np.sqrt(2*np.log(played_rounds)/batches_per_dataset[dataset_name])
            self._upper_confidence_index[dataset_name] = val
            if val > best_action_value:
                best_action_value = val
                best_action_idx = i
        
        weights = torch.zeros(len(self.auxiliary_dataset_names))
        weights[best_action_idx] = 1

        dataloader.batch_sampler.update_weights_and_distribution(weights)

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()
        target_dataloader = self.get_target_dataloader(self.target_dataset)

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        # UCB1 specific variables
        batches_per_dataset = {dataset_name: 1 for dataset_name in self.auxiliary_dataset_names}
        self._upper_confidence_index = {dataset_name: None for dataset_name in self.auxiliary_dataset_names}
        
        if args.log_samples_per_dataset:
            samples_seen_per_dataset = {}
        else:
            samples_seen_per_dataset = None

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Initialize weights if needed
        assert(self.args.weight_initialization_samples > 0), "Empirical means must be precomputed for UCB1"
        self._initialize_weights(train_dataloader, target_dataloader, model)

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        self.state.trial_name = self.hp_name(trial) if self.hp_name is not None else None
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):            
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                sample_datasets = inputs.pop("dataset_name")
                assert(all([s == sample_datasets[0] for s in sample_datasets]))
                batch_dataset = sample_datasets[0]
                if args.log_samples_per_dataset:
                    if batch_dataset not in samples_seen_per_dataset:
                        samples_seen_per_dataset[batch_dataset] = 0
                    samples_seen_per_dataset[batch_dataset] += len(sample_datasets)
                    batches_per_dataset[batch_dataset] += 1

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        tr_loss_step, grad_step = self.training_step(
                                                    model,
                                                    inputs,
                                                    batch_dataset,
                                                    scale_by_similarities=self.args.loss_scaling,
                                                    return_grads=True
                                                    )
                else:
                    tr_loss_step, grad_step = self.training_step(
                                                model,
                                                inputs,
                                                batch_dataset,
                                                scale_by_similarities=self.args.loss_scaling,
                                                return_grads=True
                                                )

                self._update_grad(batch_dataset, grad_step)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    
                    if (step + 1) % args.target_training_frequency == 0:
                        # Train on full training set before gradient update
                        for step, inputs in enumerate(target_dataloader):
                            tr_loss_step = self.training_step(
                                                        model,
                                                        inputs,
                                                        batch_dataset,
                                                        scale_by_similarities=False,
                                                        return_grads=False
                                                        )
                            tr_loss += tr_loss_step

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and not self.deepspeed:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and not self.deepspeed:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, samples_seen_per_dataset)

                    target_grad = self._calculate_grad(model, target_dataloader).detach()
                    if self.args.offload_grads:
                        target_grad = target_grad.cpu()
                        
                    self._update_grad_similarity(target_grad)
                    self._clear_grads()
                    # if self.args.weighted_batch_sampling:
                    #     self._update_dataloader_weights(train_dataloader)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.args.weighted_batch_sampling:
                    self._update_dataloader_weights(train_dataloader, batches_per_dataset, (step+1) % args.gradient_accumulation_steps)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval, samples_seen_per_dataset)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()

        # Check if we've run eval
        if self.state.global_step > args.eval_steps:
            train_loss = self._total_loss_scalar / self.state.global_step
        else:
            # if not, do so manually
            self.manual_eval_and_save(model, trial, ignore_keys_for_eval)
            train_loss = 0.0

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    def _initialize_vars_to_log(self):
        self._vars_to_log['upper_confidence_index'] = "tensor"

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval, samples_seen_per_dataset):
        if self.control.should_log:

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs['step'] = self.state.global_step
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()
            if samples_seen_per_dataset is not None:
                logs["samples_seen_per_dataset"] = {k: v for k,v in samples_seen_per_dataset.items()}
            for var,t in self._vars_to_log.items():
                if t == "scalar":
                    logs[var] = {k:v for k,v in getattr(self, f"_{var}").items()}
                elif t == "tensor":
                    logs[var] = {k:v.item() for k,v in getattr(self, f"_{var}").items()}
            # logs['upper_confidence_index'] = {k: v.item() for k,v in self._upper_confidence_index.items()}
            # logs['gradient_similarities'] = {k: v.item() for k,v in self._similarities.items()}

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)