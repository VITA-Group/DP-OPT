import numpy as np
from collections import defaultdict
from utils.dln_data import init_dataset
from datasets import load_dataset, concatenate_datasets


def sample_demos(dataset, num_demos, label_words, make_text_target=True,
                 return_subset=False, rng=np.random, input_key='sentence', 
                 balance=False, poisson=False):
    """
    Return: 
        - selected_demos: Tuple[List, List]
        - Or selected_demos, subset (a subset of dataset)
    """
    if poisson:
        choice = lambda D, k: poisson_subsample(D, k, rng)
    else:
        choice = lambda D, k: rng.choice(D, k, replace=False)
    if balance:
        smp_by_cls = defaultdict(list)
        for idx, data in enumerate(dataset):
            smp_by_cls[data['label']].append(idx)
        if num_demos < len(smp_by_cls):
            print(f'fail to split {num_demos} into {len(smp_by_cls)} classes. Randomly select some classes.')
            num_demo_by_class = 1
            _sel_classes = choice(list(smp_by_cls.keys()), num_demos)
            smp_by_cls = {c: smp_by_cls[c] for c in _sel_classes}
        else:
            num_demo_by_class = num_demos // len(smp_by_cls)
        subset = []
        for cls in smp_by_cls:
            smp_idxs = choice(smp_by_cls[cls], num_demo_by_class)
            subset.extend([dataset[i] for i in smp_idxs])
    else:
        subset = choice(dataset, num_demos)
    selected_demos = [[], []]  # input, output
    for data in subset:
        # print(f"data['label']: {data['label']}")
        selected_demos[0].append(data[input_key])
        if make_text_target:
            selected_demos[1].append(label_words[data['label']])
        else:
            selected_demos[1].append(data['label'])
    if return_subset:
        return selected_demos, subset
    else:
        return selected_demos


def poisson_subsample(D, k, rng):
    """
    Subsample k samples from dataset D using Poisson sampling.
    
    Parameters:
    - D (list): The dataset from which to sample.
    - k (int): The number of samples to draw.
    
    Returns:
    - list: A subsample of the dataset D.
    """
    n = len(D)
    if n == 0 or k == 0:
        return []
    
    # Calculate the Poisson sampling probability for each item
    p = k / n
    
    # Sample each item independently with probability p
    subsample = [item for item in D if rng.rand() < p]
    
    return subsample


def get_dataset(name, holdout_ratio, test_ratio=1., rng=np.random, adhoc_full_test=False):
    if name == 'sst2':
        raw_dataset = load_dataset('glue', name)
        label_words = ['negative', 'positive']
        data_splits = ['train', 'validation']
    elif name == 'sst2_priv':
        raw_dataset = load_dataset("zoharli/sst2_priv", revision='eps8')
        sst2_dataset = load_dataset('glue', 'sst2')
        raw_dataset['validation'] = sst2_dataset['validation']
        label_words = ['negative', 'positive']
        data_splits = ['train', 'validation']
    elif name in ("mpqa", "trec", "subj"):
        n_test = None
        dln_ds, output_classes, val_examples = init_dataset(name, 42, "./data", n_test=n_test)
        raw_dataset = dln_ds.get_hf_data()
        label_words = output_classes.protos
        data_splits = ['train', 'test']
    elif name in ("mpqa_priv", "trec_priv"):
        base_name = name.split('_')[0]
        n_test = None
        dln_ds, output_classes, val_examples = init_dataset(base_name, 42, "./data", n_test=n_test)
        raw_dataset = dln_ds.get_hf_data()
        raw_dataset['train'] = load_dataset(f"vita-group/{name}")['train']
        label_words = output_classes.protos
        data_splits = ['train', 'test']
    elif name == 'disaster':
        dln_ds, output_classes, val_examples = init_dataset(name, 42, "./data")
        dln_ds.transform_label({label_word: f"{i}" for i, label_word in enumerate(['Not Relevant', 'Relevant'])})
        raw_dataset = dln_ds.get_hf_data()
        label_words = output_classes.protos
        data_splits = ['train', 'test']
    elif name == 'disaster_priv':
        base_name = name.split('_')[0]
        dln_ds, output_classes, val_examples = init_dataset(base_name, 42, "./data")
        dln_ds.transform_label({label_word: f"{i}" for i, label_word in enumerate(['Not Relevant', 'Relevant'])})
        raw_dataset = dln_ds.get_hf_data()
        raw_dataset['train'] = load_dataset(f"vita-group/{name}")['train']
        label_words = output_classes.protos
        data_splits = ['train', 'test']
    else:
        raise NotImplementedError(f"data: {name}")

    train_dataset_size = len(raw_dataset['train'])
    holdout_indicies = rng.choice(train_dataset_size, int(train_dataset_size * holdout_ratio), replace=False)
    dataset = {'holdout': []}
    for split in data_splits:
        dataset[split] = []
        for idx, data in enumerate(raw_dataset[split]):
            if name == "mnli":
                data["sentence"] = f"premise: {data['premise']} hypothesis: {data['hypothesis']}"
            if name == "qnli":
                data["sentence"] = f"question: {data['question']} sentence: {data['sentence']}"
            if name == "qqp":
                data["sentence"] = f"question1: {data['question1']} question2: {data['question2']}"
            if split == 'train' and idx in holdout_indicies:
                dataset['holdout'].append(data)
            else:
                dataset[split].append(data)
    if 'validation' not in data_splits:
        assert 'test' in data_splits, f"Not found val set: {data_splits}"
        dataset['validation'] = dataset['test']
        del dataset['test']
    if test_ratio < 1.:
        val_subset = rng.choice(dataset['validation'], int(len(dataset['validation']) * test_ratio), replace=False)
        if not adhoc_full_test:
            dataset['validation'] = val_subset
    return dataset, label_words


