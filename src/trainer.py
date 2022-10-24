from typing import Optional, Dict, Union, Any, Tuple, List, NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from transformers import Seq2SeqTrainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.modeling_utils import unwrap_model
from transformers.utils import is_datasets_available, ModelOutput, logging
from transformers.trainer_pt_utils import IterableDatasetShard, find_batch_size, nested_numpify
from transformers.trainer_utils import seed_worker, has_length, denumpify_detensorize, EvalPrediction
from transformers.deepspeed import deepspeed_init

if is_datasets_available():
    import datasets

from data.data_utils import MTCLDataCollator

logger = logging.get_logger(__name__)

class EvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    score_candidates: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    score_gts: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]

class MTCLSeq2SeqTrainer(Seq2SeqTrainer):
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
        
        # ALON: Get custom collate_fn
        data_collator = MTCLDataCollator(
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

        # ALON: Get custom collate_fn
        data_collator = MTCLDataCollator(
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

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            score_candidates=all_score_cand,
            score_gts=all_score_gt,
            metrics=metrics,
            num_samples=num_samples
            )
