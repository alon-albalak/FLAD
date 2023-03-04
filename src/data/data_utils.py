from dataclasses import dataclass
from typing import Optional, Any, Union, Dict, List, Iterator
import logging
import math
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data.sampler import BatchSampler, RandomSampler
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from .dataset_readers import get_datasets

logger = logging.getLogger(__name__)

def get_train_val_datasets(training_args, target_dataset_args, data_args):
    # If only training with auxiliary data, use the auxiliary dataset for validation
    if training_args.train_strategy == "auxiliary_only":
        train_dataset = get_datasets(data_args.auxiliary_dataset, split="train",
                max_samples=data_args.max_samples_per_auxiliary_dataset, return_as_dict=True,
                include_T0_eval=data_args.include_T0_eval)
        validation_dataset = get_datasets(data_args.auxiliary_dataset, split="validation",
                return_as_dict=False, include_T0_eval=data_args.include_T0_eval)

    # If training with auxiliary and target data, use the target dataset for validation
    elif training_args.train_strategy == "auxiliary_and_target":
        train_dataset = get_datasets(data_args.auxiliary_dataset, split="train",
                max_samples=data_args.max_samples_per_auxiliary_dataset, return_as_dict=True,
                include_T0_eval=data_args.include_T0_eval, target_dataset=data_args.target_dataset)
        validation_dataset = get_datasets(data_args.target_dataset, split="validation",
                            target_dataset_args=target_dataset_args, return_as_dict=False)
        # If using Exp3 or UCB1, keep target dataset separate from auxiliary dataset
        if training_args.gradient_directed:
            target_dataset = get_datasets(data_args.target_dataset, split="train",
                                target_dataset_args=target_dataset_args, return_as_dict=False)
        # otherwise, get the target dataset and combine it with the auxiliary data
        else:
            target_dataset = get_datasets(data_args.target_dataset, split="train",
                                target_dataset_args=target_dataset_args, return_as_dict=True)
            train_dataset.update(target_dataset)
    # If training directly on target data
    else:
        train_dataset = get_datasets(data_args.target_dataset, split="train",
                            target_dataset_args=target_dataset_args, return_as_dict=False)
        validation_dataset = get_datasets(data_args.target_dataset, split="validation",
                            target_dataset_args=target_dataset_args, return_as_dict=False)

    if training_args.gradient_directed:
        return train_dataset, validation_dataset, target_dataset
    return train_dataset, validation_dataset

def get_test_dataset(target_dataset_args, data_args):
    return get_datasets(data_args.target_dataset, split="test",
        target_dataset_args=target_dataset_args, return_as_dict=False)

class DatasetWithTemplate(torch.utils.data.dataset.Dataset):
    """
    Class for holding HF Datasets and PromptSource Templates together
    """
    def __init__(self, dataset, tokenizer, include_answer_choices=True, add_special_tokens=True):
        super().__init__()
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.include_answer_choices = include_answer_choices
        self.add_special_tokens = add_special_tokens
        self.index_map = np.arange(len(dataset))
        self.start = None
        self.end = None
        self.dataset_name = dataset.name

    def __len__(self):
        if self.start is not None and self.end is not None:
            return self.end-self.start
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.include_answer_choices:
            return self.get_item_w_answer_choices(sample)
        else:
            return self.get_item(sample)

    def get_item(self, sample):
        template_checks = 0
        target_str = "<NO LABEL>"

        # cycle through templates until we get one that works
        while target_str == "<NO LABEL>" and template_checks < 10:
            template = np.random.choice(self.dataset.templates)
            # some samples fail ungracefully for specific templates
            try:
                input_target_str = template.apply(sample)
            except:
                logger.warn(f"Could not find suitable template for {self.dataset.name}")
                if "idx" in sample:
                    logger.warn(f"Fail on idx: {sample['idx']}")
                continue
            if len(input_target_str) == 2:
                input_str, target_str = input_target_str
                if target_str == "":
                    target_str = "<NO LABEL>"
            else:
                input_str = "<NO INPUT>"
                target_str = "<NO LABEL>"
            template_checks += 1

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
            input_ids = self.tokenizer(input_str, return_tensors="pt", truncation=True).input_ids.squeeze(0)
        target_ids = self.tokenizer(target_str, return_tensors="pt", truncation=True).input_ids.squeeze(0)
        return self.dataset.name, input_ids, target_ids

    def get_item_w_answer_choices(self, sample):
        template_checks=0
        target_str=None
        # cycle through templates until we get one that works
        while target_str is None and template_checks < 10:
            # some samples fail ungracefully for specific templates
            template = np.random.choice(self.dataset.templates)
            try:
                input_str, target_str = template.apply(sample)
            except ValueError:
                template_checks += 1
                continue
        if template_checks == 10:
            raise ValueError(f"Cannot find appropriate templates for sample: {sample}")

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
        return self.dataset.name, input_ids, target_ids, answer_choices_ids, label, idx

    def shuffle_indices(self):
        self.rng.shuffle(self.index_map)

    def initialize_iterable(self, seed, start=None, end=None):
        """Convert the Mapped dataset to an IterableDataset"""
        if start is not None and end is not None and \
            self.start is None and self.end is None:
            
            self.start = math.ceil(len(self.dataset)*start/100)
            self.end = min(math.ceil(len(self.dataset)*end/100), len(self.dataset))
            self.index_map = self.index_map[self.start:self.end]

        self.rng = np.random.default_rng(seed)
        self._restart()

    def __iter__(self):
        self.shuffle_indices()
        self.cur_idx = -1

    def _restart(self):
        self.__iter__()

    def __next__(self):
        self.cur_idx += 1
        if self.cur_idx == self.__len__():
            self._restart()
            self.cur_idx += 1
        return self.__getitem__(int(self.index_map[self.cur_idx]))

class FLADWeightedMapDataset(torch.utils.data.dataset.ConcatDataset):
    """Simple wrapper for a concatenated dataset"""
    def __init__(self,datasets, weights: Optional[List[float]] = None):
        super(FLADWeightedMapDataset, self).__init__(datasets.values())
        self.dataset_names = list(datasets.keys())
        self.weights = weights

class FLADBatchSampler(torch.utils.data.BatchSampler):
    """
    Class used to group DatasetWithTemplate samples into batches.
    Individual batches will be entirely made up of samples from only 1 dataset
    """
    def __init__(
        self,
        dataset: torch.utils.data.dataset.Dataset,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        drop_last: bool = False,
        ) -> None:
        
        self.batch_size=batch_size
        self.generator=generator
        self.drop_last=drop_last
        self.epoch=0
        self._datasets, self._dataset_idxs = [], {}
        assert(isinstance(dataset, torch.utils.data.dataset.ConcatDataset))
        cur_idx = 0
        for size, d in zip(dataset.cumulative_sizes, dataset.datasets):

            # if using batch sampler in distributed setting
            #   need to use different offsets and artificially set dataset size
            if isinstance(d, torch.utils.data.dataset.Subset):
                dataset_name = d.dataset.dataset.name
                self._dataset_idxs[dataset_name] = [idx+cur_idx for idx in d.indices]
                size = cur_idx+len(d.dataset)
            else:
                dataset_name = d.dataset.name
                self._dataset_idxs[dataset_name] = [idx for idx in range(cur_idx,size)]
            self._datasets.append(dataset_name)
            cur_idx = size

        self._reset(generator)
        self.num_batches = len(self._batch_datasets)

    def _reset(self, generator):
        # Use built-in torch samplers to randomize and batch data from each dataset
        self._samplers = {dataset: BatchSampler(RandomSampler(self._dataset_idxs[dataset], generator=generator), \
                batch_size=self.batch_size, drop_last=self.drop_last)\
            for dataset in self._datasets}
        # super complicated way to make a flattened list of dataset names corresponding to batches
        self._batch_datasets = [thing for l in [[dataset]*len(sampler) for dataset,sampler in self._samplers.items()] for thing in l]
        # convert samplers into iterables
        self._samplers = {dataset: iter(sampler) for dataset, sampler in self._samplers.items()}

    def __iter__(self) -> Iterator[int]:
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed+self.epoch)
        else:
            generator = self.generator
        
        batch_idxs = torch.randperm(self.num_batches, generator=generator).tolist()

        for idx in batch_idxs:
            batch_dataset = self._batch_datasets[idx]
            dataset_indices = next(self._samplers[batch_dataset])
            dataset_indices = [self._dataset_idxs[batch_dataset][dataset_idx] for dataset_idx in dataset_indices]
            yield dataset_indices
        self._reset(generator)

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int) -> None:
        self.epoch=epoch

class FLADWeightedBatchSampler(FLADBatchSampler):
    """
    Class used to group DatasetWithTemplate samples into batches.
    Individual batches will be entirely made up of samples from only 1 dataset.
    Batches will by sampled from datasets according to a softmax probability distribution
        defined by self._weights
    """
    def __init__(
        self,
        dataset: torch.utils.data.dataset.Dataset,
        batch_size: int,
        generator: Optional[torch.Generator] = None,
        drop_last: bool = False,

        ) -> None:

        self.batch_size=batch_size
        self.generator=generator
        self.drop_last=drop_last
        self._datasets, self._dataset_idxs = [], {}
        assert(isinstance(dataset, torch.utils.data.dataset.ConcatDataset))
        cur_idx = 0
        for size, d in zip(dataset.cumulative_sizes, dataset.datasets):
            # TODO: if needed to work with multiple workers, split up idxs here by worker number

            # if using batch sampler in distributed setting
            #   need to use different offsets and artificially set dataset size
            if isinstance(d, torch.utils.data.dataset.Subset):
                dataset_name = d.dataset.dataset.name
                self._dataset_idxs[dataset_name] = [idx+cur_idx for idx in d.indices]
                size = cur_idx+len(d.dataset)
            else:
                dataset_name = d.dataset.name
                self._dataset_idxs[dataset_name] = [idx for idx in range(cur_idx,size)]
            self._datasets.append(dataset_name)
            cur_idx = size

        self.epochs={dataset_name: 0 for dataset_name in self._datasets}

        self._reset(generator)
        self.num_batches = len(self._batch_datasets)

        self.weights = dataset.weights
        self.update_distribution()
        self.all_done = False

    def _reset(self, generator):
        # Use built-in torch samplers to randomize and batch data from each dataset
        self._samplers = {dataset: BatchSampler(RandomSampler(self._dataset_idxs[dataset], generator=generator), \
                batch_size=self.batch_size, drop_last=self.drop_last)\
            for dataset in self._datasets}
        # super complicated way to make a flattened list of dataset names corresponding to batches
        self._batch_datasets = [thing for l in [[dataset]*len(sampler) for dataset,sampler in self._samplers.items()] for thing in l]
        # convert samplers into iterables
        self._samplers = {dataset: iter(sampler) for dataset, sampler in self._samplers.items()}

    def _reset_single(self, generator, dataset):
        sampler = BatchSampler(RandomSampler(self._dataset_idxs[dataset], generator=generator), \
                    batch_size=self.batch_size, drop_last=self.drop_last)
        self._samplers[dataset] = iter(sampler)

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @weights.setter
    def weights(self, weights) -> torch.Tensor:
        if isinstance(weights, list):
            self._weights = torch.nn.functional.softmax(torch.tensor(weights), dim=0)
        elif isinstance(weights, torch.Tensor):
            self._weights = torch.nn.functional.softmax(weights, dim=0)
        elif weights is None:
            self._weights = torch.tensor([1/len(self._datasets) for _ in self._datasets])
        else:
            raise TypeError("weights must be a list or Tensor")

    @property
    def distribution(self) -> torch.distributions.distribution.Distribution:
        return self._distribution

    @distribution.setter
    def distribution(self, dist):
        self._distribution = dist

    @property
    def all_done(self) -> bool:
        return self._all_done

    @all_done.setter
    def all_done(self, done):
        self._all_done = done

    def update_distribution(self):
        assert(self.weights is not None)
        self.distribution = torch.distributions.categorical.Categorical(probs=self.weights)

    def update_weights_and_distribution(self, weights, threshold=None):
        if threshold:
            weights = torch.nn.functional.threshold(weights, threshold, float("-Inf"))
            if not torch.any(weights > float("-Inf")):
                self.all_done = True
                return
        self.weights = weights
        self.update_distribution()

    def _infinite_iterator(self):
        while True:

            # check for manual exit
            if self.all_done:
                break

            # sample a dataset
            batch_dataset = self._datasets[self.distribution.sample()]
            
            try:
                dataset_indices = next(self._samplers[batch_dataset])
            except StopIteration:
                # if no batches left for this dataset, reset
                self.iterate_epoch(batch_dataset)
                generator = torch.Generator()
                generator.manual_seed(self.generator.seed() + self.epochs[batch_dataset])
                self._reset_single(generator, batch_dataset)
                dataset_indices = next(self._samplers[batch_dataset])

            dataset_indices = [self._dataset_idxs[batch_dataset][dataset_idx] for dataset_idx in dataset_indices]
            # print(f"Dataset: {batch_dataset} Idx: {dataset_indices}")
            yield dataset_indices

    def __iter__(self) -> Iterator[int]:
        return self._infinite_iterator()

    def __len__(self) -> int:
        return self.num_batches

    def set_epoch(self, epoch: int, dataset_name: str) -> None:
        self.epochs[dataset_name]=epoch

    def iterate_epoch(self, dataset_name: str) -> None:
        self.epochs[dataset_name] += 1


class FLADDistributedBatchSampler(
    torch.utils.data.distributed.DistributedSampler,
    torch.utils.data.sampler.BatchSampler
):
    def __init__(
            self,
            dataset: torch.utils.data.dataset.Dataset,
            num_replicas: Optional[int] = None,
            rank: Optional[int] = None,
            shuffle: bool = True,
            seed: int = 0,
            batch_size: int = 8
        ) -> None:
        
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.dataset = self._split_dataset(dataset)
        self.num_samples = math.ceil(len(dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        self.batch_size = batch_size

    def _split_dataset(self, datasets) -> torch.utils.data.dataset.Dataset:
        per_replica = int(math.ceil(100/float(self.num_replicas)))
        rank_start = self.rank*per_replica
        rank_end = min(rank_start+per_replica, 100)

        rank_datasets = []
        for d in datasets.datasets:
            d_start = math.ceil(len(d)*rank_start/100)
            d_end = min(math.ceil(len(d)*rank_end/100), len(d))
            rank_datasets.append(torch.utils.data.dataset.Subset(d, range(d_start,d_end)))
        return torch.utils.data.dataset.ConcatDataset(rank_datasets)

    def __iter__(self) -> Iterator:

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch + self.rank)

        batch_sampler = FLADBatchSampler(
            self.dataset,
            batch_size=self.batch_size,
            generator=generator,
            )
        return iter(batch_sampler)
    
    def __len__(self) -> int:
        return (self.num_samples + self.batch_size - 1) // self.batch_size

class FLADWeightedIterableDataset(torch.utils.data.IterableDataset):
    """
    Dataset class used to wrap multiple DatasetWithTemplate's
    Each DatasetWithTemplate will be sampled according to the probability 
            distribution defined by self.weights
    """


    def __init__(
        self,
        datasets: Dict[str, DatasetWithTemplate],
        weights: Optional[List[float]] = None,
        seed = None
    ) -> None:

        self.datasets = datasets
        self.ordered_dataset_names = list(datasets.keys())
        self.weights = weights
        self.seed = seed
        self.generator = self._initialize_generator(seed)

        if not isinstance(self.weights, (torch.Tensor, list)):
            raise TypeError("weights should be a list or tensor of floats, "
                            f"but got weights={self.weights}")

        self.update_distribution()

    def _initialize_datasets(self, seed, worker_start=None, worker_end=None):
        """Convert Mapped datasets into IterableDatasets"""
        for dataset in self.datasets.values():
            dataset.initialize_iterable(seed, worker_start, worker_end)

    def _initialize_generator(self, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    @property
    def weights(self) -> torch.Tensor:
        return self._weights

    @weights.setter
    def weights(self, weights) -> torch.Tensor:
        if isinstance(weights, list):
            self._weights = torch.tensor(weights)
        elif isinstance(weights, torch.Tensor):
            self._weights = weights
        elif weights is None:
            self._weights = torch.tensor([1/len(self.ordered_dataset_names) for _ in self.ordered_dataset_names])
        else:
            raise TypeError("weights must be a list or Tensor")

    @property
    def distribution(self) -> torch.distributions.distribution.Distribution:
        return self._distribution

    @distribution.setter
    def distribution(self, dist):
        self._distribution = dist

    def update_distribution(self):
        assert(self.weights is not None)
        self.distribution = torch.distributions.categorical.Categorical(probs=self.weights)

    def _infinite_iterator(self):
        while True:
            dataset = self.ordered_dataset_names[self.distribution.sample()]
            yield next(self.datasets[dataset])

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            worker_generator = self._initialize_generator(self.seed + worker_id)
            seed = worker_generator.seed()
            per_worker = int(math.ceil(100 / float(num_workers)))
            worker_start = worker_id*per_worker
            worker_end = min(worker_start + per_worker, 100)
            self._initialize_datasets(seed, worker_start, worker_end)
        else:
            worker_id = 0
            self._initialize_datasets(self.generator.seed())

        return self._infinite_iterator()

@dataclass
class FLADDataCollator:
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
            dataset_name, input_ids, target_ids, answer_choices_ids, labels, idx = zip(*batch)
        else:
            dataset_name, input_ids, target_ids = zip(*batch)

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attention_mask = (input_ids != self.tokenizer.pad_token_id).float()  # [bs, max_seq_len]
        
        output_batch = {
            "dataset_name": dataset_name,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        if self.pretrain:
            target_ids = torch.nn.utils.rnn.pad_sequence(target_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            lm_labels = target_ids + -100 * (target_ids == self.tokenizer.pad_token_id).long()  # [bs, max_seq_len]
            output_batch.update({"target_ids": lm_labels})
            if (
                target_ids is not None
                and self.model is not None
                and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
            ):
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=target_ids)
                output_batch.update({"decoder_input_ids": decoder_input_ids})

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
