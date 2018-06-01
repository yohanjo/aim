import os
import pickle
import time
import logging
import math
import sys
import torch
import json
import numpy as np
from csv_utils import *
from constants import *
from collections import defaultdict
from torch.autograd import Variable as V
from torch import LongTensor as LT
from torch import FloatTensor as FT
from zipfile import ZipFile, ZIP_DEFLATED

def long_var(l, requires_grad=False):
    """Long tensor variable, with cuda if applicable."""
    tensor = LT(l)
    if torch.cuda.is_available(): tensor = tensor.cuda()
    var = V(tensor, requires_grad=requires_grad)
    return var

def float_var(l, requires_grad=False):
    """Float tensor variable, with cuda if applicable."""
    tensor = FT(l)
    if torch.cuda.is_available(): tensor = tensor.cuda()
    var = V(tensor, requires_grad=requires_grad)
    return var

def cuda_var(tensor):
    if torch.cuda.is_available(): return tensor.cuda()
    return tensor

def zero_var(*sizes):
    """Zero tensor variable, with cuda if applicable."""
    tensor = torch.zeros(sizes)
    if torch.cuda.is_available(): tensor = tensor.cuda()
    var = V(tensor)
    return var

def empty_var():
    vec = torch.randn(0,0)
    if torch.cuda.is_available(): vec = vec.cuda()
    return vec

def mean(l, default):
    """Get mean if l is not empty, otherwise returns default."""
    if len(l) == 0: return default
    else: return sum(l) / len(l)


def get_logger(path):
    """Returns a logger that logs into the console and a file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
                
    for handler in [logging.FileHandler(path, mode='w'), 
                    logging.StreamHandler()]:
        logger.addHandler(handler)

    return logger

def prepare_directory(path):
    """Creates a directory if not exists."""
    if not os.path.exists(path):
        print("Making directory: {}".format(path))
        os.makedirs(path, exist_ok=True)

def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def load_data_info():
    """Loads and returns the data split information."""
    data_split = dict()
    for row in iter_csv_header(SPLIT_PATH):
        data_split[row["SeqId"]] = row

    return data_split


def load_data(vocab_size, splits=["train", "val", "test"], 
              max_seq_len=sys.maxsize, 
              max_n_seqs=sys.maxsize, domain=None):
    """Loads data from JSON files and generates sequences."""
    print("Loading data: {} (vocab={}, max_n_seqs={}, max_seq_len={})".format(
            ", ".join(splits), vocab_size, max_n_seqs, max_seq_len))

    # Load the sequence IDs of the domain to load
    id_list = None
    if domain is not None:
        data_split = load_data_info()
        id_list = [seq_id for seq_id, info in data_split.items() \
                   if info["Domain"] == domain]

    # Load data
    seqs = defaultdict(list)
    for split in splits:
        delta = defaultdict(int)
        for line in open(DATA_DIR + "/{}_{}.json".format(split, vocab_size)):
            skip_line = False
            msgs = dict()
            op_msgs = []
            for idx, node_str in \
                    enumerate(line.strip().split("\t")[:max_seq_len]):
                node = json.loads(node_str)
                if idx == 0 and id_list is not None \
                            and node["post_name"] not in id_list:
                    skip_line = True
                    break

                msgs[node["post_name"]] = node
                if node["is_op"]: op_msgs.append(node)
            if skip_line: continue

            seq = [op_msgs[0]]
            post_name_set = set([op_msgs[0]["post_name"]])
            for msg in op_msgs[1:]:
                sub_seq = []
                while msg["post_name"] not in post_name_set:
                    sub_seq.insert(0, msg)
                    post_name_set.add(msg["post_name"])
                    delta[msg['delta']] += 1
                    if msg["parent_id"] not in msgs: break
                    else: msg = msgs[msg["parent_id"]]
                seq.extend(sub_seq)
            seqs[split].append(seq)
            if len(seqs[split]) >= max_n_seqs: break

        print("[{}] n_seqs = {} (median seq_len: {}), "
              "delta = [0: {}, 1: {} ({:.1f}%)]".format(
            split, len(seqs[split]), 
            np.median([len(seq) for seq in seqs[split]]),
            delta[0], delta[1], 
            (delta[1] / (delta[0] + delta[1]) * 100)))

    # Load vocabulary
    voca = open("{}/voca_{}.txt".format(DATA_DIR, vocab_size))\
            .read().strip().split("\n")

    return seqs, voca

def load_data_pickle_basic(vocab_size, splits=["train", "val", "test"], 
                          max_seq_len=sys.maxsize, max_n_seqs=sys.maxsize, 
                          domains=["all"]):
    print("Loading data from pickle: {} (vocab={}, domain={})".format(
            ", ".join(splits), vocab_size, domains))

    seqs = defaultdict(list)
    for split in splits:
        for domain in domains:
            path = "{}/V40000-basic/V{}_{}_{}.p".format(
                            DATA_DIR, vocab_size, domain, split)
            domain_seqs = pickle_load(path)

            # Filter
            if max_n_seqs < len(domain_seqs):
                domain_seqs = domain_seqs[:max_n_seqs]
            if max_seq_len < sys.maxsize:
                for idx, seq in enumerate(domain_seqs):
                    if len(seq) > max_seq_len:
                        domain_seqs[idx] = seq[:max_seq_len]

            seqs[split].extend(domain_seqs)


    # Load vocabulary
    voca = open("{}/voca_{}.txt".format(DATA_DIR, vocab_size))\
            .read().strip().split("\n")

    return seqs, voca




def load_data_pickle(vocab_size, splits=["train", "val", "test"], 
                     max_seq_len=sys.maxsize, max_n_seqs=sys.maxsize, 
                     domains=["all"]):
    print("Loading data from pickle: {} (vocab={}, domain={})".format(
            ", ".join(splits), vocab_size, domains))

    seqs = defaultdict(list)
    for split in splits:
        for domain in domains:
            path = "{}/V{}_{}_{}.p".format(DATA_DIR, vocab_size, domain, split)
            domain_seqs = pickle_load(path)

            # Filter
            if max_n_seqs < len(domain_seqs):
                domain_seqs = domain_seqs[:max_n_seqs]
            if max_seq_len < sys.maxsize:
                for idx, seq in enumerate(domain_seqs):
                    if len(seq) > max_seq_len:
                        domain_seqs[idx] = seq[:max_seq_len]

            seqs[split].extend(domain_seqs)


    # Load vocabulary
    voca = open("{}/voca_{}.txt".format(DATA_DIR, vocab_size))\
            .read().strip().split("\n")

    return seqs, voca

def load_data_pairs(vocab_size, splits=["train", "val", "test"], 
                     max_seq_len=sys.maxsize, max_n_seqs=sys.maxsize, 
                     domains=["all"]):
    print("Loading paired data from pickle: {} (vocab={}, domain={})".format(
            ", ".join(splits), vocab_size, domains))

    seqs = defaultdict(list)
    for split in splits:
        for domain in domains:
            path = "{}/V{}_{}_{}.p".format(DATA_DIR, vocab_size, domain, split)
            domain_seqs = pickle_load(path)

            # Filter
            if max_n_seqs < len(domain_seqs):
                domain_seqs = domain_seqs[:max_n_seqs]
            if max_seq_len < sys.maxsize:
                for idx, (op_node, comments) in enumerate(domain_seqs):
                    if len(comments) > max_seq_len:
                        domain_seqs[idx] = (op_node, comments[:max_seq_len])

            seqs[split].extend(domain_seqs)


    # Load vocabulary
    voca = open("{}/voca_{}.txt".format(DATA_DIR, vocab_size))\
            .read().strip().split("\n")

    return seqs, voca



def load_data_pairs_(vocab_size, splits=["train", "val", "test"], 
                     max_seq_len=sys.maxsize, max_n_seqs=sys.maxsize, 
                     domains=["all"]):
    split_seqs, voca = load_data_pickle(vocab_size, splits=splits,
                                  max_seq_len=max_seq_len, 
                                  max_n_seqs=max_n_seqs,
                                  domains=domains)

    split_pairs = defaultdict(list)
    for split, seqs in split_seqs.items():
        for seq in seqs:
            comments = []
            comment_ids = set()
            pair = (seq[0], comments)
            for node in seq[1:]:
                if not node["is_op"]: continue
                if node["parent_id"] not in node_dict: continue
                parent = node_dict[node["parent_id"]]
                if parent["is_op"]: continue

                if "delta" not in parent or parent["delta"] == 0:
                    parent["delta"] = node["delta"]
                if parent["post_name"] not in comment_ids:
                    comments.append(parent)
                    comment_ids.add(parent["post_name"])
            if len(comments) > 0:
                split_pairs[split].append(pair)

    return split_pairs, voca
 
 
def save_data_pickle(seqs, vocab_size, domain=None):
    for split in seqs.keys():
        pickle_dump(seqs[split],
                    "{}/V{}_{}_{}.p".format(
                        DATA_DIR, vocab_size,
                        (domain if domain is not None else "all"),
                        split))

def iter_node(seqs, split_exists=True):
    """Iterates over all nodes in the sequences."""
    if split_exists:
        for split in seqs.keys():
            for seq in seqs[split]:
                for node in seq:
                    yield node
    else:
        for seq in seqs:
            for node in seq:
                yield node

def pickle_dump(obj, path):
    """Platform-safe pickle.dump. Identical to pickle.dump, except that
    pickle.dump may raise an error on Mac for a big file (>2GB)."""
    max_bytes = 2 ** 31 - 1
    out_bytes = pickle.dumps(obj)
    n_bytes = len(out_bytes)
    with open(path, "wb") as f:
        for idx in range(0, n_bytes, max_bytes):
            f.write(out_bytes[idx:(idx + max_bytes)])


def pickle_load(path):
    """Platform-safe pickle.load. Identical to pickle.load, except that
    pickle.load may raise an error on Mac for a big file."""
    max_bytes = 2 ** 31 - 1
    input_size = os.path.getsize(path)
    obj_bytes = bytearray(0)
    with open(path, "rb") as f:
        for _ in range(0, input_size, max_bytes):
            obj_bytes += f.read(max_bytes)
    return pickle.loads(obj_bytes)

def zip_and_delete(file_path, zip_path):
    with ZipFile(zip_path, 'w') as f:
        f.write(file_path, os.path.basename(file_path), ZIP_DEFLATED)
    os.remove(file_path)
