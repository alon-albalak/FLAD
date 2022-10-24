from dataclasses import dataclass
from json import decoder
from typing import Optional, Any, Union
import numpy as np
import torch
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy

class DatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, tokenizer, include_answer_choices=True, add_special_tokens=True):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.include_answer_choices = include_answer_choices
        self.add_special_tokens = add_special_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        template = np.random.choice(self.dataset.templates)
        sample = self.dataset[idx]
        if self.include_answer_choices:
            return self.get_item_w_answer_choices(sample, template)
        else:
            return self.get_item(sample, template)


    def get_item(self, sample, template):
        input_target_str = template.apply(sample)
        if len(input_target_str) == 2:
            input_str, target_str = input_target_str
            if target_str == "":
                target_str = "<NO LABEL>"
        else:
            input_str = "<NO INPUT>"
            target_str = "<NO LABEL>"
        input_ids = self.tokenizer(input_str, return_tensors="pt", truncation=True).input_ids.squeeze(0)
        target_ids = self.tokenizer(target_str, return_tensors="pt", truncation=True).input_ids.squeeze(0)
        return input_ids, target_ids

    def get_item_w_answer_choices(self, sample, template):
        input_str, target_str = template.apply(sample)
        answer_choices = template.get_answer_choices_list(sample)
        if isinstance(input_str, list):
            input_ids = torch.cat(
                [
                    self.tokenizer(
                        input_field, return_tensors="pt", truncation=True, add_special_tokens=False
                    ).input_ids.squeeze(0)
                    for input_field in input_str[:-1]
                ]
                + [
                    self.tokenizer(
                        input_str[-1], return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
                    ).input_ids.squeeze(0)
                ]
            )
        else:
            input_ids = self.tokenizer(
                input_str, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
            ).input_ids.squeeze(0)
        target_ids = self.tokenizer(
            target_str, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
        ).input_ids.squeeze(0)
        answer_choices_ids = [
            self.tokenizer(
                answer_choice, return_tensors="pt", truncation=True, add_special_tokens=self.add_special_tokens
            ).input_ids.squeeze(0)
            for answer_choice in answer_choices
        ]
        if sample['label'] is not None:
            label = torch.LongTensor([sample["label"]])
        else:
            label = None
        idx = torch.LongTensor([sample["idx"]])
        return input_ids, target_ids, answer_choices_ids, label, idx

@dataclass
class MTCLDataCollator:
    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"
    pretrain: bool = False

    def __call__(self, batch, return_tensors=None):
        if return_tensors is None:
            return_tensors=self.return_tensors

        if not self.pretrain:
            input_ids, target_ids, answer_choices_ids, labels, idx = zip(*batch)
        else:
            input_ids, target_ids = zip(*batch)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()  # [bs, max_seq_len]
        
        output_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.pretrain:
            target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            lm_labels = target_ids + -100 * (target_ids == self.tokenizer.pad_token_id).long()  # [bs, max_seq_len]
            output_batch.update({"target_ids": lm_labels})
            # decoder_input_ids = torch.cat(
            #     [torch.zeros_like(lm_labels[:, :1]), target_ids[:, :-1]], dim=1
            # )  # [bs, max_seq_len]
            # decoder_attention_mask = (decoder_input_ids == decoder_input_ids).float()
            
            # ALON: HF Style
            if (
                target_ids is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
            ):
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=target_ids)
                output_batch.update({"decoder_input_ids": decoder_input_ids})


            # output_batch = {
            #     "input_ids": input_ids,
            #     "attention_mask": attention_mask,
            #     "decoder_input_ids":decoder_input_ids,
            #     "target_ids": lm_labels,
            # }

        else:
            flat_answer_choice_ids = [choice for list_choices in answer_choices_ids for choice in list_choices]
            num_choice = [len(list_choices) for list_choices in answer_choices_ids]
            if max(num_choice) != min(num_choice):
                raise NotImplementedError("The collate_fn is not implmented for variable number of choices")
            flat_answer_choices_ids = torch.nn.utils.rnn.pad_sequence(
                flat_answer_choice_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
            )
            answer_choices_ids = flat_answer_choices_ids.view(len(answer_choices_ids), max(num_choice), -1).contiguous()
            labels = torch.cat(labels)
            idx = torch.cat(idx)
            output_batch.update(
                {
                    "answer_choices_ids": answer_choices_ids,
                    "labels": labels,
                    "idx": idx,
                }
            )

        return output_batch
