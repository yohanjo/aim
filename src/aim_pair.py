import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable as V
from encoders import InteractEncoder
import re
import os
import argparse
import sys
import csv
import time
from time import strftime, localtime
from random import shuffle
from helper import *
from constants import *
from collections import defaultdict
from itertools import combinations
from sklearn.metrics import roc_auc_score


class G:
    """Global variables."""
    args = None
    log_prefix = None
    logger = None
    start_time = None

    domains = None
    domains_cd = None

    seqs = None
    seqs_cd = None
    voca = None

    delta_criterion = None
    rank_criterion = None

    encoder = None


def main():
    parse_args()
    prepare_directories()
    prepare_logging()
    prepare_criteria()

    # Load train, val, test, and test_cd (cross-domain test)
    G.seqs, G.voca = load_data_pairs(
                        G.args.vocab_size, ["train", "val", "test"], 
                        G.args.max_seq_len, G.args.max_n_seqs,
                        G.domains)
    G.seqs_cd, _ = load_data_pairs(
                        G.args.vocab_size, ["test"], 
                        G.args.max_seq_len, G.args.max_n_seqs,
                        G.domains_cd)
    G.seqs["test_cd"] = G.seqs_cd["test"]

    # Create the model
    G.encoder = InteractEncoder(G.args.in_emb_dim,
                                G.args.hidden_dim, G.args.hidden_comb,
                                G.args.sent_emb, G.args.dropout,
                                G.args.attention, G.args.interaction)
    if torch.cuda.is_available():
        G.encoder = G.encoder.cuda()
    
    prepare_optimizer()

    # Train and test
    run_epochs(G.seqs["train"], G.seqs["val"], 
               G.seqs["test"], G.seqs_cd["test"])

def parse_args():
    parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-domains', dest='domains', type=str, nargs="+",
                        help="Domains of data to load. (domains7: the in-"
                             "domains used in the paper)")
    parser.add_argument('-domains_cd', dest='domains_cd', type=str, nargs="+",
                        help="Domains for cross-domain evaluationtion. "
                             "(domains_cd: the cross-domains used in the "
                             "paper)")
    parser.add_argument('-vocab_size', dest='vocab_size', type=int, 
                        default=VOCAB_SIZE, help="Maximum vocabulary size.")
    parser.add_argument('-max_seq_len', dest='max_seq_len', type=int, 
                        default=sys.maxsize,
                        help="Sequences longer than max length are trimmed.")
    parser.add_argument('-max_n_seqs', dest='max_n_seqs', type=int, 
                        default=sys.maxsize,
                        help="Max number of sequences to include in the data.")
    parser.add_argument('-n_epochs', dest='n_epochs', default="auto",
                        help="Number of epochs. If auto, max is set to 15.")

    # Encoding
    parser.add_argument("-attention", dest="attention", action="store_true",
                        default=False, help="Use attention?")
    parser.add_argument("-sent_emb", dest="sent_emb", type=str,
                        default="infersent",
                        help="Sentence embedding method. {infersent, glove}")
    parser.add_argument("-in_emb_dim", dest="in_emb_dim", type=int, default=64,
                        help="Output dim of sentence encoder.")
    parser.add_argument('-dropout', dest='dropout', type=float, 
                        default=0.5, help="Dropout ratio.")
    parser.add_argument("-interaction", dest="interaction", type=str,
                        default="prod", 
                        help="Interaction embedding method. (prod: inner "
                             "product, m1_..._mL: feed-forward network with "
                             "mi being the num of nodes on the i-th layer)")

    # Prediction 
    parser.add_argument("-hidden_comb", dest="hidden_comb", type=str, nargs="+",
                        default=["cur", "max"], 
                        help="Prediction input. (max: MAX, cur: HSENT, wsdim: "
                             "WDO, tfidf.A.n: TFIDF with a feedforward net "
                             "with n output nodes)")
    parser.add_argument('-hidden_dim', dest='hidden_dim', type=int, default=32,
                        help="Dim of the last layer before classifier.")

    # Optimization
    parser.add_argument('-rank_loss_margin', dest='rank_loss_margin', 
                        type=float, default=0.5, 
                        help="Margin for the margin rank loss.")
    parser.add_argument('-rank_loss_weight', dest='rank_loss_weight', 
                        type=float, default=1, help="Ranking loss weight.")
    parser.add_argument('-learning_rate', dest='learning_rate', default="auto",
                        help="Learning rate. If 'auto', set to 1e-4 for Adam, "
                             "1e-2 for SGD")
    parser.add_argument("-optimizer", dest="optimizer", type=str,
                        default="adamax", help="Optimizer {adam, adamax, sgd}")
    parser.add_argument("-batch_size", dest="batch_size", type=int,
                        default=1, help="Minibatch size.")
    G.args = parser.parse_args()

    # Default
    if G.args.learning_rate == "auto":
        if G.args.optimizer == "adam": G.args.learning_rate = 1e-4
        elif G.args.optimizer == "sgd": G.args.learning_rate = 1e-2
        elif G.args.optimizer == "adamax": G.args.learning_rate = 0.002
    else:
        G.args.learning_rate = float(G.args.learning_rate)

    if G.args.n_epochs != "auto":
        G.args.n_epochs = int(G.args.n_epochs)

    G.domains = DOMAINS7 if "domains7" in G.args.domains else G.args.domains
    G.domains_cd = DOMAINS_CD if "domains_cd" in G.args.domains_cd \
                              else G.args.domains_cd

    if "cur" in G.args.hidden_comb:
        G.args.hidden_dim = G.args.hidden_dim
    else:
        G.args.hidden_dim = \
                len(set(["max", "mean"]) & set(G.args.hidden_comb))

def prepare_criteria():
    G.delta_criterion = nn.BCELoss()  # Binary cross-entropy loss
    G.rank_criterion = \
            nn.MarginRankingLoss(G.args.rank_loss_margin) # Margin ranking loss

def prepare_directories():
    """Make sure all directories exist."""
    prepare_directory(MODEL_DIR)
    prepare_directory(LOG_DIR)
    prepare_directory(PLOT_DIR)

def prepare_logging():
    """Prepare log prefix and logger."""
    G.log_prefix = \
        "{}-Inter-VO{}-DO{}-AT{}-SE{}-ID{}-IT{}"\
        "-HC{}-HD{}-RM{}-RW{}-DR{}-OT{}-LR{}-BS{}".format(
            strftime("%Y%m%d_%H%M%S", localtime()), G.args.vocab_size,
            "_".join(sorted(G.args.domains)),
            G.args.attention, G.args.sent_emb,
            G.args.in_emb_dim, 
            G.args.interaction,
            "_".join(sorted(G.args.hidden_comb)),
            G.args.hidden_dim,
            G.args.rank_loss_margin, G.args.rank_loss_weight,
            G.args.dropout,
            G.args.optimizer, G.args.learning_rate,
            G.args.batch_size)
 
    pattern = re.sub(".*Inter-", "", G.log_prefix)
    for filename in os.listdir(LOG_DIR):
        if pattern in filename or pattern.replace("HD1-", "HD0-") in filename:
            print("A log with the same setting exists: {}".format(filename))
            sys.exit()
    G.logger = get_logger("{}/{}.txt".format(LOG_DIR, G.log_prefix))

def prepare_optimizer():
    params_to_opt = []
    params_to_opt.extend(G.encoder.parameters())

    if G.args.optimizer == "adam":
        G.optimizer = optim.Adam(params_to_opt, lr=G.args.learning_rate)
    elif G.args.optimizer == "sgd":
        G.optimizer = optim.SGD(params_to_opt, lr=G.args.learning_rate)
    elif G.args.optimizer == "adamax":
        G.optimizer = optim.Adamax(params_to_opt, lr=G.args.learning_rate)
    else:
        raise Exception("Invalid optimizer {}".format(G.args.optimizer))

def set_training_mode(training):
    """Sets train/eval mode for the encoder.
    
    Args:
        training: True if training mode, False if eval mode.
    """
    G.encoder.train(training)

def train(seqs):
    set_training_mode(True)
    delta_loss, rank_loss, delta_conf_mat, delta_auc = \
        run(seqs, do_optimization=True)
    return delta_loss, rank_loss, delta_conf_mat, delta_auc


def test(seqs):
    set_training_mode(False)
    delta_loss, rank_loss, delta_conf_mat, delta_auc = \
        run(seqs)
    return delta_loss, rank_loss, delta_conf_mat, delta_auc


def run(seqs, do_optimization=False):
    """Runs the model on the given sequences and optionally optimizes the model.

    Args:
        seqs: Data sequences.
        do_optimization: Do optimization?

    Returns:
        delta_loss: Average delta loss.
        rank_loss: Average rank loss.
        delta_conf_mat: Confusion matrix of delta prediction.
        delta_auc: AUC score of delta.
    """
    start_time = time.time() # Epoch start time

    delta_pairs = [] # List of (true delta, pred delta)
    delta_losses = []
    rank_losses = []
    # Confusion matrix (True delta x Predicted delta)
    delta_conf_mat = np.zeros((2, 2), dtype=np.uint32)  

    batch_delta_pairs = []
    for seq_idx, (op_node, comments) in enumerate(seqs):

        # Encode OP's first post
        op_embs, attn_weights = G.encoder.encode_op_post(op_node)
        op_node["attn_weights"] = attn_weights.data[0, :, 0].tolist()

        for comment in comments:
            # Predict Delta
            pred_delta, interact, attn_interact = \
                    G.encoder(comment, op_embs, attn_weights)
            true_delta = float_var([[[comment['delta']]]])
            batch_delta_pairs.append({"true_delta": true_delta, 
                                      "pred_delta": pred_delta})
            comment["pred_delta"] = pred_delta.data[0, 0, 0]
            comment["interact"] = interact
            comment["attn_interact"] = attn_interact


        # Compute losses
        if ((seq_idx + 1) % G.args.batch_size == 0 or seq_idx + 1 == len(seqs)):
            # Delta losses
            batch_delta_losses = []
            for pair in batch_delta_pairs:
                t_delta = pair["true_delta"].data[0, 0, 0]
                p_delta = pair["pred_delta"].data[0, 0, 0]
                delta_pairs.append((t_delta, p_delta))

                loss = G.delta_criterion(pair["pred_delta"], 
                                         pair["true_delta"])
                batch_delta_losses.append(loss)

                # Confusion matrix
                delta_conf_mat[int(round(t_delta)), int(round(p_delta))] += 1
            batch_delta_loss_avg = mean(batch_delta_losses, float_var([0]))
            delta_losses.append(batch_delta_loss_avg.data[0])

            # Ranking losses
            batch_rank_losses = []
            for pair1, pair2 in combinations(batch_delta_pairs, 2):
                d1 = pair1["true_delta"].data[0, 0, 0]
                d2 = pair2["true_delta"].data[0, 0, 0]
                if d1 == d2: continue
                y = float_var([[[1]]]) if d1 > d2 else float_var([[[-1]]])
                batch_rank_losses.append(G.args.rank_loss_weight * \
                        G.rank_criterion(pair1["pred_delta"], 
                                         pair2["pred_delta"], y))
            batch_rank_loss_avg = mean(batch_rank_losses, float_var([0]))
            rank_losses.append(batch_rank_loss_avg.data[0])

            # Optimize the model
            if do_optimization:
                G.optimizer.zero_grad()
                loss = batch_delta_loss_avg + batch_rank_loss_avg
                if loss.data[0] != 0:
                    loss.backward()
                    G.optimizer.step()

            batch_delta_pairs = []

        G.logger.info("{} {} ".format(seq_idx + 1, 
            time_since(start_time, float(seq_idx + 1) / len(seqs))))

    # Calculate average loss
    delta_loss_avg = mean(delta_losses, np.nan)
    rank_loss_avg = mean(rank_losses, np.nan)
    delta_auc = roc_auc_score([pair[0] for pair in delta_pairs],
                              [pair[1] for pair in delta_pairs])

    return delta_loss_avg, rank_loss_avg, \
           delta_conf_mat, delta_auc 


def run_epochs(train_seqs, val_seqs, test_seqs, test_cd_seqs):
    """Run the model on the train/test data for epochs.

    Args:
        train_seqs: Train sequences.
        val_seqs: Validation sequences.
        test_seqs: Test sequences.
        test_cd_seqs: Test sequences for cross-domain.
    """
    G.start_time = time.time()

    print("Running {} epochs...".format(G.args.n_epochs))
    max_n_epochs = G.args.n_epochs if G.args.n_epochs != "auto" else 15

    delta_losses_train = []
    rank_losses_train = []
    aucs_train = []

    delta_losses_val = []
    rank_losses_val = []
    aucs_val = []

    delta_losses_test = []
    rank_losses_test = []
    aucs_test = []

    delta_losses_test_cd = []
    rank_losses_test_cd = []
    aucs_test_cd = []

    for epoch in range(1, max_n_epochs + 1):
        # Train
        if train_seqs is not None:
            #shuffle(train_seqs) # Shuffle training sequences
            delta_loss_train, rank_loss_train, delta_conf_mat_train, \
                    delta_auc_train = train(train_seqs)
            delta_losses_train.append(delta_loss_train)
            rank_losses_train.append(rank_loss_train)
            aucs_train.append(delta_auc_train)

        # Val
        if val_seqs is not None:
            delta_loss_val, rank_loss_val, \
                    delta_conf_mat_val, delta_auc_val = test(val_seqs)
            delta_losses_val.append(delta_loss_val)
            rank_losses_val.append(rank_loss_val)
            aucs_val.append(delta_auc_val)

        # Test
        if test_seqs is not None:
            delta_loss_test, rank_loss_test, \
                    delta_conf_mat_test, delta_auc_test = test(test_seqs)
            delta_losses_test.append(delta_loss_test)
            rank_losses_test.append(rank_loss_test)
            aucs_test.append(delta_auc_test)

        # Test (corss-domain)
        if test_cd_seqs is not None:
            delta_loss_test_cd, rank_loss_test_cd, \
                delta_conf_mat_test_cd, delta_auc_test_cd = test(test_cd_seqs)
            delta_losses_test_cd.append(delta_loss_test_cd)
            rank_losses_test_cd.append(rank_loss_test_cd)
            aucs_test_cd.append(delta_auc_test_cd)

        # Plot losses
        show_plot([[[delta_losses_train, delta_losses_val, 
                     delta_losses_test, delta_losses_test_cd], 
                    ["Train", "Val", "Test", "Test CD"], 
                    "Delta loss", "Loss"],
                   [[rank_losses_train, rank_losses_val, 
                     rank_losses_test, rank_losses_test_cd],
                    ["Train", "Val", "Test", "Test CD"], 
                    "Ranking loss", "Loss"],
                   [[aucs_train, aucs_val, aucs_test, aucs_test_cd],
                    ["Train", "Val", "Test", "Test CD"], 
                    "AUC score", "AUC"]],
                  "{}/{}-plots.pdf".format(PLOT_DIR, G.log_prefix))


        # Log
        G.logger.info(
            'Epoch: {} ({}%), Time: {} '
            'train_delta_loss={:.4f}, train_rank_loss={:.4f}, '
            'train_delta_conf={}, train_delta_auc={:.4f}, '
            'val_delta_loss={:.4f}, val_rank_loss={:.4f}, '
            'val_delta_conf={}, val_delta_auc={:.4f}, '
            'test_delta_loss={:.4f}, test_rank_loss={:.4f}, '
            'test_delta_conf={}, test_delta_auc={:.4f}, '
            'test_cd_delta_loss={:.4f}, test_cd_rank_loss={:.4f}, '
            'test_cd_delta_conf={}, test_cd_delta_auc={:.4f}'.format(
            epoch, epoch / (max_n_epochs * 1.0) * 100,
            time_since(G.start_time, epoch / (max_n_epochs * 1.0)), 
            delta_losses_train[-1] if len(delta_losses_train) > 0 else np.nan, 
            rank_losses_train[-1] if len(rank_losses_train) > 0 else np.nan, 
            delta_conf_mat_train.tolist(), delta_auc_train,
            delta_losses_val[-1] if len(delta_losses_val) > 0 else np.nan,
            rank_losses_val[-1] if len(rank_losses_val) > 0 else np.nan, 
            delta_conf_mat_val.tolist(), delta_auc_val,
            delta_losses_test[-1] if len(delta_losses_test) > 0 else np.nan,
            rank_losses_test[-1] if len(rank_losses_test) > 0 else np.nan, 
            delta_conf_mat_test.tolist(), delta_auc_test,
            delta_losses_test_cd[-1] if len(delta_losses_test_cd) > 0 else np.nan,
            rank_losses_test_cd[-1] if len(rank_losses_test_cd) > 0 else np.nan, 
            delta_conf_mat_test_cd.tolist(), delta_auc_test_cd))

        # Save the model if the current model is best
        if len(aucs_val) == 1 or delta_auc_val > max(aucs_val[:-1]):
            print("Saving the model...")
            torch.save(G.encoder, 
                       "{}/{}.p".format(MODEL_DIR, G.log_prefix))

            print("Printing the sequences...")
            print_seqs(epoch)

        # Stopping condition
        if G.args.n_epochs == "auto" and epoch >= 10 and epoch % 5 == 0 and \
                    np.mean(aucs_val[-5:]) < np.mean(aucs_val[-10:-5]): 
            break

    # Zip and delete the sequence file because it's too big
    zip_and_delete("{}/{}-seqs.csv".format(LOG_DIR, G.log_prefix),
                   "{}/{}-seqs.zip".format(LOG_DIR, G.log_prefix))

    # Learning rate decay
    if G.args.optimizer == "adamax":
        for param in G.optimizer.param_groups:
            param["lr"] *= 0.95


def show_plot(data, plot_path):
    """Plot the input data into a single PDF file.
    Args:
        data: [[[points1, ...], [legend1, ...], title, ylabel], ...]
        plot_path: Path to the output PDF file.
    """
    style=["-o", "-^", "-s", "-x", "-+", "-D"]
    with PdfPages(plot_path) as pdf:
        for points_list, legend, title, ylabel in data:
            n_points = len(points_list[0])
            fig, ax = plt.subplots()
            plt.title(title)
            for p, points in enumerate(points_list):
                plt.plot(points, style[p])
            ax.set_xlabel("Epoch")
            ax.set_xticks(range(n_points))
            label_ticks = [0] + [(n_points - 1) - int(i * n_points / 5) \
                                 for i in range(5)]
            ax.set_xticklabels([x + 1 if x in label_ticks else "" \
                                for x in range(n_points)])
            ax.set_ylabel(ylabel)
            plt.legend(legend, loc="best")
            pdf.savefig()
            plt.close(fig)

def print_seqs(epoch):
    s_time = time.time()
    out_path = "{}/{}-seqs.csv".format(LOG_DIR, G.log_prefix)

    info = dict()
    header = ["Split", "SeqId", "NodeId", "ParentId", "SentenceNo", #"Text",
              "IsOP", "Delta", "PredDelta", "Attention",
              "Interact", "AttnInteract"]

    with open(out_path, "w") as f:
        out_csv = csv.writer(f)
        out_csv.writerow(header)
        for split, seqs in G.seqs.items():
            for op_node, comments in seqs:
                seq_id = op_node["post_name"]
                for node in [op_node] + list(comments):
                    attn_weights = node.get("attn_weights", None)
                    interact = node.get("interact", None)
                    attn_interact = node.get("attn_interact", None)

                    for idx, sentence in enumerate(
                            filter(lambda s: len(s) > 0, node["word_vec"])):
                        sent_id = (node["post_name"], idx)

                        info[sent_id] = {
                                "Split": split,
                                "SeqId": seq_id,
                                "NodeId": node["post_name"],
                                "ParentId": node["parent_id"],
                                "SentenceNo": idx,
                                "IsOP": node["is_op"],
                                "Delta": node["delta"]}

                        info[sent_id]["Attention"] = \
                                attn_weights[idx] if attn_weights is not None \
                                                  else ""
                        info[sent_id]["PredDelta"] = node.get("pred_delta", "")

                        info[sent_id]["Interact"] = interact[idx].tolist() \
                            if split == "test" and interact is not None \
                            else ""
                        info[sent_id]["AttnInteract"] = \
                            attn_interact[idx].tolist() \
                            if split == "test" and attn_interact is not None \
                            else ""

                        out_csv.writerow([info[sent_id][k] for k in header])

    print("{:.1f} secs".format(time.time() - s_time))


def show_attention(attentions, img_path):
    """
    Args:
        attentions: [[(word_idx, 1 x attn matrix), ...], ...]
    """
    # Get an attention matrix for the last decoded post
    words = []
    attn_mat_list = []
    for word_idx, attn in attentions[-1][:10]:
        words.append(G.voca[word_idx])
        attn_mat_list.append(attn)
    attn_mat = torch.cat(attn_mat_list, dim=0)

    # Set up figure with colorbar
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attn_mat.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_yticklabels([''] + words)
    ax.set_xticklabels([''] + [i for i in range(attn_mat.size()[1])])

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.set_xlabel("Prev posts")
    ax.set_ylabel("Words to decode")
    fig.savefig(img_path)
    plt.close(fig)


if __name__ == '__main__':
    main()
