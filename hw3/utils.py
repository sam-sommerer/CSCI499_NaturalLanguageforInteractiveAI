import re
import torch
import numpy as np
from collections import Counter


def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []
    padded_lens = []
    inst_count = 0
    for episode in train:
        for inst, _ in episode:
            inst = preprocess_string(inst)
            padded_len = 2  # start/end
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
            padded_lens.append(padded_len)
    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    action_lens = []

    for i, episode in enumerate(train):
        action_lens.append(0)
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)
            action_lens[i] += 1
    actions_to_index = {a: i + 3 for i, a in enumerate(actions)}
    actions_to_index["<SOS>"] = 0
    actions_to_index["<EOS>"] = 1
    actions_to_index["<pad>"] = 2
    targets_to_index = {t: i + 3 for i, t in enumerate(targets)}
    targets_to_index["<SOS>"] = 0
    targets_to_index["<EOS>"] = 1
    targets_to_index["<pad>"] = 2
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}

    avg_num_actions = sum(action_lens) // len(action_lens)

    return actions_to_index, index_to_actions, targets_to_index, index_to_targets, avg_num_actions


def prefix_match(predicted_labels, gt_labels):
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1

    seq_length = len(gt_labels)

    for i in range(seq_length):
        if predicted_labels[i] != gt_labels[i]:
            break

    pm = (1.0 / seq_length) * i

    return pm
