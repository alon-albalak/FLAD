# Much of this code is adapted from the T-FEW repository:
#   https://github.com/r-three/t-few

import os
import json
import logging
import numpy as np
from datasets import load_dataset, Dataset
from promptsource.templates import DatasetTemplates, TemplateCollection
import pandas as pd

# set logging level to INFO
logger = logging.getLogger(__name__)
logger.setLevel(20)

CACHE_DIR=os.getenv("HF_HOME", default=None)

TOMixture = [
    ("glue","mrpc"), # Paraphrase identification
    ("glue","qqp"),
    ("paws","labeled_final"),
    ("kilt_tasks", "hotpotqa"), # Closed-book QA
    ("wiki_qa", None),
    ("adversarial_qa", "dbidaf"), # Extractive QA
    ("adversarial_qa","dbert"),
    ("adversarial_qa","droberta"),
    ("duorc","SelfRC"),
    ("duorc","ParaphraseRC"),
    ("ropes",None),
    ("quoref",None),
    ("cos_e","v1.11"), # Multiple-choice QA
    ("cosmos_qa",None),
    ("dream",None),
    ("qasc",None),
    ("quail",None),
    ("quarel",None),
    ("quartz",None),
    ("sciq",None),
    ("social_i_qa",None),
    ("wiki_hop","original"),
    ("wiqa",None),
    ("amazon_polarity",None), # Sentiment
    ("app_reviews",None),
    ("imdb",None),
    ("rotten_tomatoes",None),
    ("yelp_review_full",None),
    ("common_gen",None), # Structure-to-text
    ("wiki_bio",None),
    ("cnn_dailymail","3.0.0"), # Summarization
    ("gigaword",None),
    ("multi_news",None),
    ("samsum",None),
    ("xsum",None),
    ("ag_news",None), # Topic Classification
    ("dbpedia_14",None),
    ("trec",None)
]

GPT3_eval_datasets = [ # T0p datasets in addition to T0Mixture
    ("ai2_arc", "ARC-Challenge"), # Closed-book QA
    ("ai2_arc","ARC-Easy"),
    ("trivia_qa","unfiltered"),
    ("web_questions",None),
    ("squad_v2",None), # Extractive QA
    ("openbookqa","main"), # Multiple-choice QA
    ("race","high"),
    ("race","middle"),
    ("piqa",None),
    ("hellaswag",None) # Sentence Completion
]

T0ppMixture = [ # T0pp datasets in addition to T0Mixture and T0p datasets
    ("super_glue","wsc.fixed"), # Coreference Resolution
    ("super_glue","record"), # Extractive QA
    ("super_glue","boolq"), # Multiple-choice QA
    ("super_glue","multirc"),
    ("super_glue","copa"), # Sentence Completion
    ("super_glue","wic"), # Word Sense Disambiguation
]

P3Mixture_split_map={
    # Validation
    ("asset", "simplification"): "validation",
    ("fever","v2.0"): "validation",
    ("jfleg", None): "validation",
    ("glue","mnli_matched"): "validation",
    ("glue","mnli_mismatched"): "validation",
    ('mc_taco', None): "validation",
    ('squad_adversarial', 'AddSent'): "validation",
    ('turk', None): "validation",
    ('wino_bias', 'type1_anti'): "validation",
    ('wino_bias', 'type1_pro'): "validation",
    ('wino_bias', 'type2_anti'): "validation",
    ('wino_bias', 'type2_pro'): "validation",
    ('xquad', 'xquad.en'): "validation",
    ('xquad_r', 'en'): "validation",
    # Test
    ("climate_fever", None): "test",
    ("craffel/openai_lambada",None): "test",
    ("crows_pairs",None): "test",
    ("glue","ax"): "test",
    ('openai_humaneval', None): "test",
    ('squadshifts', 'amazon'): "test",
    ('squadshifts', 'new_wiki'): "test",
    ('squadshifts', 'nyt'): "test",
    ('super_glue', 'axb'): "test",
    ('super_glue', 'axg'): "test",
    ('winograd_wsc', 'wsc273'): "test",
    ('winograd_wsc', 'wsc285'): "test",
    # Other
    ("asset", "ratings"): "full",
    ("docred", None): "train_annotated",
}

P3SkippedDatasets=[
    ('Zaid/coqa_expanded', None),
    ('jigsaw_unintended_bias', None), # requires manual download
    ('asnq', None), # download is broken
    ("coqa", None), # temporarily not working
    ('emotion', None), # Not working
]

EvalMixture = [
    "rte",
    "h-swag",
    "copa",
    "wic",
    "winogrande",
    "cb",
    "story_cloze",
    "anli-r1",
    "anli-r2",
    "anli-r3",
    "wsc"
]

EvalMixtureFullNames=[
    ("super_glue", "rte"),
    ("hellaswag",None),
    ("super_glue", "copa"),
    ("super_glue", "wic"),
    ("winogrande", "winogrande_xl"),
    ("super_glue", "cb"),
    ("story_cloze", "2016"),
    ("anli",None),
    ("super_glue", "wsc.fixed"),
]

TEST_SET_SPLITS = {
    "rte": "validation",
    "h-swag": "validation",
    "copa": "validation",
    "wic": "validation",
    "winogrande": "validation",
    "cb": "validation",
    "story_cloze": "validation",
    "anli-r1": "test",
    "anli-r2": "test",
    "anli-r3": "test",
    "wsc": "validation"
}

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

RAFT_DATASETS=[
    "ade_corpus_v2",
    "banking_77",
    "terms_of_service",
    "tai_safety_research",
    "neurips_impact_statement_risks",
    "overruling",
    "systematic_review_inclusion",
    "one_stop_english",
    "tweet_eval_hate",
    "twitter_complaints",
    "semiconductor_org_types",
]

def get_dataset_name(name: str, subset: str):
    if subset is not None:
        canonized_name = f"{name}/{subset}"
    else:
        canonized_name = name
    return canonized_name

def get_datasets(
        dataset_or_mixture_name,
        split,
        max_samples=None,
        template_idx=-1,
        target_dataset_args=None,
        return_as_dict=False
    ):
    assert(split in ['train','validation','test'])
    if dataset_or_mixture_name == "T0Mixture":
        return get_T0MixtureDatasets(split, max_samples, return_as_dict)
    elif dataset_or_mixture_name == "P3":
        return get_P3MixtureDatasets(split, max_samples, return_as_dict)
    elif dataset_or_mixture_name in EvalMixture:
        return get_eval_dataset(dataset_or_mixture_name, split, target_dataset_args, return_as_dict)
    elif dataset_or_mixture_name in RAFT_DATASETS:
        return get_raft_dataset(dataset_or_mixture_name, split, target_dataset_args, return_as_dict)
    raise ValueError(f"Unknown dataset or mixture: {dataset_or_mixture_name}")

def get_T0MixtureDatasets(split, max_samples=None, return_as_dict=True):
    """
    T0MixtureDatasets creates a separate dataset for each dataset in the mixture
    """
    datasets = {} if return_as_dict else []
    for name, subset in TOMixture:
        dataset = load_dataset(name, subset, split=split, cache_dir=CACHE_DIR)
        if max_samples:
            dataset = Dataset.from_dict(dataset[:max_samples])
        templates = [template for id, template in DatasetTemplates(name, subset).templates.items()]
        dataset.templates = templates
        dataset.name = get_dataset_name(name, subset)

        if return_as_dict:
            datasets[get_dataset_name(name, subset)] = dataset
        else:
            datasets.append(dataset)


        logger.info(f"Loaded dataset {name}/{subset} with {len(templates)} templates")
        assert(len(templates) > 0), "No templates"
    return datasets

def get_P3MixtureDatasets(split, max_samples = None, return_as_dict=True):
    """
    P3Mixture creates a separate dataset for each dataset in
        P3 that is NOT in the evaluation mixture
    """
    TC = TemplateCollection()
    datasets = {} if return_as_dict else []
    for k, v in TC.datasets_templates.items():
        tmp_split = split
        name, subset = k
        logger.info(f"Loading ({name},{subset})")

        # Skip evaluation datasets
        if (name in EvalMixture) or k in EvalMixtureFullNames:
            continue
        
        # Skip some datasets for various reasons
        if k in P3SkippedDatasets:
            continue

        if k in P3Mixture_split_map:
            tmp_split = P3Mixture_split_map[k]

        if max_samples:
            try:
                dataset = load_dataset(name, subset, split=f"{tmp_split}[:{max_samples}]", cache_dir=CACHE_DIR)
            except:
                logger.warn(f"Failed to load max samples of {k}. Loading full dataset.")
                dataset = load_dataset(name, subset, split=tmp_split, cache_dir=CACHE_DIR)
                dataset = Dataset.from_dict(dataset[:max_samples])

        else:
            dataset = load_dataset(name, subset, split=tmp_split, cache_dir=CACHE_DIR)

        templates = [template for id, template in v.templates.items()]
        dataset.templates = templates
        dataset.name = get_dataset_name(name, subset)

        if return_as_dict:
            datasets[get_dataset_name(name, subset)] = dataset
        else:
            datasets.append(dataset)

        logger.info(f"Loaded dataset {name}/{subset} with {len(templates)} templates")
        assert(len(templates) > 0), "No templates"
    return datasets

def get_eval_dataset(name, split, target_dataset_args, return_as_dict=False):
    READERS={
        "rte": RTEReader,
        "h-swag": HSwagReader,
        "copa": COPAReader,
        "wic": WiCReader,
        "winogrande": WinograndeReader,
        "cb": CBReader,
        "story_cloze": StoryClozeReader,
        "anli-r1": ANLIR1Reader,
        "anli-r2": ANLIR2Reader,
        "anli-r3": ANLIR3Reader,
        "wsc": WSCFixedReader,
    }
    reader = READERS[name](target_dataset_args)
    # if testing, use full dataset
    if split == "test":
        dataset = reader.read_orig_dataset(TEST_SET_SPLITS[name])
    elif target_dataset_args.num_shot is None:
        dataset = reader.read_orig_dataset(split)
    else:
        assert(target_dataset_args.num_shot == TEST_SET_SHOTS[name])
        dataset = reader.read_few_shot_dataset(name, target_dataset_args.num_shot,
                                    target_dataset_args.few_shot_random_seed, split)
    if isinstance(dataset, list):
        dataset = Dataset.from_list(dataset)
    templates = reader.templates
    dataset.templates = templates
    dataset.name = get_dataset_name(name, None)
    if return_as_dict:
        return {get_dataset_name(name, None): dataset}
    else:
        return dataset

def get_raft_dataset(name, split, target_dataset_args, return_as_dict=False):
    reader = RaftReader(name, target_dataset_args)
    dataset = reader.read_orig_dataset(split)
    if isinstance(dataset, list):
        dataset = Dataset.from_list(dataset)
    templates = reader.templates
    dataset.templates = templates
    dataset.name = get_dataset_name(name, None)
    if return_as_dict:
        return {get_dataset_name(name, None): dataset}
    else:
        return dataset


class BaseDatasetReader(object):
    """
    DatasetReader is responsible for reading and processing dataset
    """

    def __init__(self, dataset_stash):
        """
        :param config:
        """
        self.dataset_stash = dataset_stash
        self.templates = [template for id, template in DatasetTemplates(*self.dataset_stash).templates.items()]

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        orig_data = load_dataset(*self.dataset_stash, split=split, cache_dir=CACHE_DIR)
        return orig_data

    def read_few_shot_dataset(self, name, num_shot, few_shot_random_seed, split):
        file_dir = os.path.join("data", "few_shot", name, f"{num_shot}_shot")
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)

        file_path = os.path.join(file_dir, f"{few_shot_random_seed}_seed_{split}.jsonl")

        if os.path.exists(file_path):
            with open(file_path, "r") as fin:
                data = []
                for idx, line in enumerate(fin.readlines()):
                    data.append(json.loads(line.strip("\n")))

            return data
        else:
            orig_data = self.read_orig_dataset("train")
            train_data, validation_data = self._generate_few_shot_data(orig_data, num_shot, few_shot_random_seed)

            with open(os.path.join(file_dir, f"{few_shot_random_seed}_seed_train.jsonl"), "w+") as fout:
                for example in train_data:
                    fout.write(json.dumps(example) + "\n")
            with open(os.path.join(file_dir, f"{few_shot_random_seed}_seed_validation.jsonl"), "w+") as fout:
                for example in validation_data:
                    fout.write(json.dumps(example) + "\n")

            if split == "train":
                return train_data
            else:
                return validation_data

    def _sample_few_shot_data(self, orig_data, num_shot, few_shot_random_seed):
        saved_random_state = np.random.get_state()
        np.random.seed(few_shot_random_seed)
        orig_data = [x for x in orig_data]
        np.random.shuffle(orig_data)
        selected_data = orig_data[: num_shot]
        np.random.set_state(saved_random_state)
        return selected_data

    def _generate_few_shot_data(self, orig_data, num_shot, few_shot_random_seed):
        sampled_data = self._sample_few_shot_data(orig_data, num_shot, few_shot_random_seed)
        assert(num_shot/2 == num_shot//2),"Number of shots is not equally divisible by 2"
        train_data = sampled_data[:num_shot//2]
        validation_data = sampled_data[num_shot//2:]
        return train_data, validation_data


    def get_compute_metric(self):
        def compute_metric(accumulated):
            matching = [a == b for a, b in zip(accumulated["prediction"], accumulated["label"])]
            accuracy = sum(matching) / len(matching)
            return {"accuracy": accuracy}
        return compute_metric

class StoryClozeReader(BaseDatasetReader):
    def __init__(self, target_dataset_args):
        super().__init__(dataset_stash=("story_cloze", "2016"))

    def read_orig_dataset(self, split):
        if split == "train":
            split = "validation"
        elif split == "validation":
            split = "test"
        
        orig_data = load_dataset(*self.dataset_stash, split=split,
                    data_dir="data/story_cloze/", cache_dir=CACHE_DIR)

        orig_data = [example for example in orig_data]
        for idx, example in enumerate(orig_data):
            example["label"] = example["answer_right_ending"] - 1
            example["idx"] = idx
        return orig_data


class ANLIR1Reader(BaseDatasetReader):
    def __init__(self, target_dataset_args):
        super().__init__(dataset_stash=("anli",))

    def read_orig_dataset(self, split):
        # if split == "validation":
        #     split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r1")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class ANLIR2Reader(BaseDatasetReader):
    def __init__(self, target_dataset_args):
        super().__init__(dataset_stash=("anli",))

    def read_orig_dataset(self, split):
        # if split == "validation":
        #     split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r2")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class ANLIR3Reader(BaseDatasetReader):
    def __init__(self, target_dataset_args):
        super().__init__(dataset_stash=("anli",))

    def read_orig_dataset(self, split):
        # if split == "validation":
        #     split = "test"
        orig_data = [example for example in super().read_orig_dataset(f"{split}_r3")]
        for idx, example in enumerate(orig_data):
            example["idx"] = idx
        return orig_data


class WSCFixedReader(BaseDatasetReader):
    def __init__(self, target_dataset_args):
        super().__init__(dataset_stash=("super_glue", "wsc.fixed"))


class RTEReader(BaseDatasetReader):
    def __init__(self, target_dataset_args):
        super().__init__(dataset_stash=("super_glue", "rte"))


class HSwagReader(BaseDatasetReader):
    def __init__(self, target_dataset_args):
        super().__init__(dataset_stash=("hellaswag",))
        if target_dataset_args.change_hswag_templates:
            from promptsource.templates import Template

            name_jinja = [
                ("basic", "{{ctx}}|||{{endings [label | int()]}}"),
                (
                    "prompt 1",
                    "Can you pick the correct ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
                ),
                (
                    "prompt 2",
                    "The task is to generate the ending for the sentence: {{ctx}}|||{{answer_choices [label | int()]}}",
                ),
                ("prompt 3", "How does this sentence end? {{ctx}}|||{{answer_choices [label | int()]}}"),
                (
                    "prompt 4",
                    "From the list of endings described below, what ending makes the most sense for the sentence {{ctx}}|||{{answer_choices [label | int()]}}",
                ),
                (
                    "ctx a,b",
                    "Complete the description with an appropriate ending:\n First, {{ ctx_a.lower() }} Then, {{ ctx_b.lower() }} ...|||{{answer_choices [label | int()]}}",
                ),
                (
                    "middle",
                    "If a description of a situation begins like this: {{ ctx }}... Then how does it continue?|||{{answer_choices [label | int()]}}",
                ),
            ]

            self.templates = []
            for name, jinja in name_jinja:
                self.templates.append(
                    Template(name=name, jinja=jinja, reference="", answer_choices='{{endings | join("|||")}}')
                )

    def read_orig_dataset(self, split):
        orig_data = [example for example in super().read_orig_dataset(split)]
        has_label=False
        for idx, example in enumerate(orig_data):
            if idx==0:
                if example['label'] == "":
                    logger.warn(f"Dataset split {split} of {self.dataset_stash} has no labels, cannot be used for ")
                else:
                    has_label=True
            if has_label:
                example["label"] = int(example["label"])
            example["idx"] = idx
        return orig_data


class WiCReader(BaseDatasetReader):
    def __init__(self, target_dataset_args):
        super().__init__(dataset_stash=("super_glue", "wic"))


class COPAReader(BaseDatasetReader):
    def __init__(self, target_dataset_args):
        super().__init__(dataset_stash=("super_glue", "copa"))
        #self.templates = self.templates[:8]


class WinograndeReader(BaseDatasetReader):
    def __init__(self, target_dataset_args):
        super().__init__(dataset_stash=("winogrande", "winogrande_xl"))

    def read_orig_dataset(self, split):
        orig_data = [example for example in super().read_orig_dataset(split)]
        for idx, example in enumerate(orig_data):
            example["label"] = int(example["answer"]) - 1
            example["idx"] = idx
        return orig_data


class CBReader(BaseDatasetReader):
    def __init__(self, target_dataset_args):
        super().__init__(dataset_stash=("super_glue", "cb"))

# TODO: Adjust RAFT data to match data-loading strategy
class RaftTemplate(object):
    def __init__(self, dataset_name, target_dataset_args, answer_choices):
        with open(os.path.join(os.path.dirname(__file__), "raft_prompt_construction_settings.jsonl")) as f:
            data = [json.loads(line) for line in f]
            FIELD_ORDERING = data[0]
            INSTRUCTIONS = data[1]
        self.dataset_name = dataset_name
        self.answer_choices = answer_choices
        self.instruction = INSTRUCTIONS[self.dataset_name]
        self.fields = FIELD_ORDERING[self.dataset_name]
        self.raft_labels_in_input_string = target_dataset_args.raft_labels_in_input_string

    def apply(self, example):
        if self.raft_labels_in_input_string == "comma":
            input_str = [
                self.instruction.strip()
                + " Possible labels: "
                + ", ".join([choice for index, choice in enumerate(self.answer_choices)])
            ]
        elif self.raft_labels_in_input_string == "newline":
            input_str = [
                self.instruction.strip()
                + "\nPossible labels:\n"
                + "\n".join([str(index + 1) + ". " + choice for index, choice in enumerate(self.answer_choices)])
            ]
        else:
            input_str = [self.instruction.strip()]

        for key in example:
            if key in self.fields:
                if example[key].strip() != "":
                    input_str.append(str(key) + ": " + example[key].strip())

        if example["label"] == -1:
            target_str = "Unlabeled"
        else:
            target_str = self.answer_choices[example["label"]]
        input_str[-1] += "\nLabel:"
        return input_str, target_str

    def get_answer_choices_list(self, example):
        return self.answer_choices


class RaftReader(object):
    def __init__(self, dataset_name, target_dataset_args):
        self.target_dataset_args = target_dataset_args
        self.orig_data = load_dataset("ought/raft", name=dataset_name, cache_dir=CACHE_DIR)
        self.answer_choices = self.orig_data["train"].features["Label"].names[1:]
        if dataset_name == "banking_77" and target_dataset_args.cleaned_answer_choices_b77:
            self.answer_choices = [answer.replace("_", " ").replace(". ", " ") for answer in self.answer_choices]

        self.templates = [RaftTemplate(dataset_name, target_dataset_args, self.answer_choices)]

    # def get_train_template(self):
    #     return self.template

    # def get_eval_template(self):
    #     return self.template

    def read_orig_dataset(self, split):
        """
        Read the original dataset

        :param split: split of data
        """
        if self.target_dataset_args.raft_cross_validation:
            orig_data = [example for example in self.orig_data["train"]]
            if self.target_dataset_args.few_shot_random_seed:
                np.random.seed(self.target_dataset_args.few_shot_random_seed)
            else:
                np.random.seed(42)
            np.random.shuffle(orig_data)
            # if split == "train":
            #     orig_data = (
            #         orig_data[: self.config.raft_validation_start] + orig_data[self.config.raft_validation_start + 10 :]
            #     )
            #     assert len(orig_data) == 40
            # elif split == "validation":
            #     orig_data = orig_data[self.config.raft_validation_start : self.config.raft_validation_start + 10]
            #     assert len(orig_data) == 10
            if split == "train":
                orig_data = orig_data[:25]
            elif split == "validation":
                orig_data = orig_data[25:]
        else:
            if split == "validation":
                split = "test"
            orig_data = [example for example in self.orig_data[split]]
        for i, example in enumerate(orig_data):
            # if self.dataset_name in ['ade_corpus_v2', 'terms_of_service','overruling']:
            #     example['input'] = example['Sentence'].strip()
            # elif self.dataset_name in ['banking_77']:
            #     example['input'] = example['Query'].strip()
            # elif self.dataset_name in ['tai_safety_research']:
            #     example['input'] = 'Title : ' + example['Title'].strip() + ' ' + \
            #         'Abstract Note : ' + example['Abstract Note'].strip() + ' '+ \
            #             'Url : ' + example['Url'].strip() + ' ' + \
            #                 'Publication Year : ' + example['Publication Year'].strip() + ' '+ \
            #                     'Item Type : ' + example['Item Type'].strip() + ' ' + \
            #                         'Author : ' + example['Author'].strip() + ' '+ \
            #                             'Publication Title : '  + example['Publication Title'].strip()
            # elif self.dataset_name in ['neurips_impact_statement_risks']:
            #     example['input'] = 'Paper title : ' + example['Paper title'].strip() + ' ' + \
            #         'Paper link : ' + example['Paper link'].strip() + ' ' + \
            #             'Impact statement : ' + example['Impact statement'].strip()
            # elif self.dataset_name in ['systematic_review_inclusion']:
            #     example['input'] = 'Title : ' + example['Title'].strip() + ' ' + \
            #         'Abstract : ' + example['Abstract'].strip() + ' ' + \
            #             'Authors : ' + example['Authors'].strip() + ' ' + \
            #                 'Journal : ' + example['Journal'].strip()
            # elif self.dataset_name in ['one_stop_english']:
            #     example['input'] = example['Article'].strip()
            # elif self.dataset_name in ['tweet_eval_hate']:
            #     example['input'] = example['Tweet'].strip()
            # elif self.dataset_name in ['twitter_complaints']:
            #     example['input'] = example['Tweet text'].strip()
            # elif self.dataset_name in ['semiconductor_org_types']:
            #     example['input'] = 'Paper title : ' + example['Paper title'].strip() + \
            #         'Organization name : ' + example['Organization name'].strip()
            example["label"] = int(example["Label"]) - 1
            example["idx"] = example["ID"]
        return orig_data

    def get_compute_metric(self):
        def compute_metric(accumulated):
            data = []
            idxs = accumulated["idx"]
            predictions = accumulated["prediction"]
            # for idx, prediction in zip(idxs, predictions):
            #     data.append({"ID": idx, "Label": self.answer_choices[prediction]})
            # result_df = pd.DataFrame(data=data, columns=["ID", "Label"]).astype({"ID": int, "Label": str})
            # result_df.to_csv(self.config.dev_pred_file, index=False)
            # matching = [a == b for a, b in zip(accumulated["prediction"], accumulated["label"])]
            # accuracy = sum(matching) / len(matching)
            # return {"accuracy": accuracy}

            answer_choices = accumulated['answer_choices']
            for idx, prediction in zip(idxs, predictions):
                data.append({"ID": idx, "Label": answer_choices[prediction]})
            result_df = pd.DataFrame(data=data, columns=["ID", "Label"]).astype({"ID": int, "Label": str})
            result_df.to_csv(self.config.dev_pred_file, index=False)
            matching = [a == b for a, b in zip(accumulated["prediction"], accumulated["label"])]
            accuracy = sum(matching) / len(matching)
            return {"accuracy": accuracy}
        return compute_metric