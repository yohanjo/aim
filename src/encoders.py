"""InferSent-based msg encoder."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from helper import *
from constants import *


class InteractEncoder(nn.Module):
    """Encodes OP's post and comment, and predict delta."""

    def __init__(self, in_emb_dim, hidden_dim, hidden_comb, sent_emb_method,
                 dropout, use_attention, interaction):
        """
        Args:
            in_emb_dim: Output dim of sentence encoder.
            hidden_dim: Output dim of hidden layer.
            hidden_comb: List of input types fed to hidden layer.
                         {cur, max, mean, feat, op}
            sent_emb_method: Sentence embedding method. {glove, infersent}
            dropout: Dropout rate for input vector.
            use_attention: Use attention?
            interaction: Interaction embedding method.
        """
        super(InteractEncoder, self).__init__()
        self.hidden_comb = hidden_comb
        self.interaction = interaction
        self.use_attention = use_attention

        # Dimensionalities
        if sent_emb_method == "infersent": self.in_dim = SENT_EMB_DIM
        elif sent_emb_method == "glove": self.in_dim = GLOVE_DIM
        else: raise Exception("Invalid sentence embedding method: {}".format(
                                                        sent_emb_method))
        self.in_emb_dim = in_emb_dim
        if interaction == "prod":
            inter_emb_dim = 1
        else:
            inter_layer_dim = list(map(int, interaction.split("_")))
            if len(inter_layer_dim) > 2: raise Exception("Invalid interaction")
            self.inter_layer_dim = inter_layer_dim
            inter_emb_dim = inter_layer_dim[-1]

        # Predictor input dim
        hidden_in_dim = 0
        if "cur" in hidden_comb: hidden_in_dim += in_emb_dim
        if "max" in hidden_comb: hidden_in_dim += inter_emb_dim
        if "mean" in hidden_comb: hidden_in_dim += inter_emb_dim
        self.sent_emb_method = sent_emb_method

        # Layers
        if sent_emb_method == "glove": 
            self.word_emb_layer = self.init_word_emb_layer()
        self.sent_dropout = nn.Dropout(dropout)
        self.sent_encoder = nn.GRU(self.in_dim, in_emb_dim)
        self.attn_layer1 = nn.Linear(in_emb_dim, 1)
        # Interaction layer
        if interaction != "prod":
            if len(inter_layer_dim) == 1:
                self.inter_layer = nn.Sequential(
                        nn.Linear(in_emb_dim * 2, inter_layer_dim[0]), 
                        nn.Tanh())
            else:
                self.inter_layer = nn.Sequential(
                        nn.Linear(in_emb_dim * 2, inter_layer_dim[0]), 
                        nn.Tanh(), 
                        nn.Linear(inter_layer_dim[0], inter_layer_dim[1]), 
                        nn.Tanh())

        # Tfidf layer
        self.tfidf_layer = None
        self.tfidf_type = None
        tfidf_in_dim = 0
        tfidf_out_dim = 0
        for feature in hidden_comb:
            if not feature.startswith("tfidf."): continue
            _, tfidf_type, tfidf_emb_dim = feature.split(".") # E.g., tfidf.A.3
            tfidf_in_dim += VOCAB_SIZE
            tfidf_out_dim += int(tfidf_emb_dim)
            self.tfidf_type = tfidf_type
            break

        if tfidf_in_dim > 0:
            self.tfidf_layer = nn.Sequential(
                    nn.Linear(tfidf_in_dim, tfidf_out_dim), nn.ReLU())

        # Word-similarity layer
        self.wdsim_type = None
        wdsim_dim = 0
        for feature in hidden_comb:
            if not feature.startswith("wdsim."): continue
            _, wdsim_type = feature.split(".") # E.g., wdsim.A
            wdsim_dim += 4
            self.wdsim_type = wdsim_type
            break

        # Hidden and prediction layers
        #if hidden_dim > 1:
        new_hidden_in_dim = hidden_in_dim + \
                            (tfidf_out_dim if self.tfidf_type == "B" \
                                           else 0) + \
                            (wdsim_dim if self.wdsim_type == "B" else 0)
        new_hidden_dim = hidden_dim + \
                         (tfidf_out_dim if self.tfidf_type == "A" \
                                        else 0) + \
                         (wdsim_dim if self.wdsim_type == "A" else 0)

        self.hidden_layer = nn.Sequential(
            nn.Linear(new_hidden_in_dim, hidden_dim), nn.ReLU())
        self.predict_layer = nn.Sequential(
            nn.Linear(new_hidden_dim, 1), nn.Sigmoid())
        #else:
        #    self.hidden_layer = nn.Dropout(0) # Identity
        #    self.predict_layer = nn.Sequential(
        #        nn.Linear(hidden_in_dim + tfidf_out_dim, 1), nn.Sigmoid())


    def init_word_emb_layer(self):
        """Initializes a GloVe word embedding layer."""
        glove = pickle_load(GLOVE_NUMPY_PATH)
        word_emb_layer = nn.Embedding(glove.shape[0], glove.shape[1])
        word_emb_layer.weight.data.copy_(torch.from_numpy(glove))
        return word_emb_layer

    def get_sent_embs(self, node):
        """Embeds the given node's sentences. Uses either GloVe embeddings or
        InferSent directly depending on the configuration."""
        if self.sent_emb_method == "glove":
            sents = []
            for sentence in node["word_vec"]:
                if len(sentence) == 0: continue
                word_embs = self.word_emb_layer(
                        long_var([sentence])) # 1 x n_words x 300
                sents.append(word_embs[0].sum(dim=0)) # (300,)
            sents = torch.stack(sents, dim=0) # l1 x 300
        else: # InferSent
            sents = float_var(node["sentence_emb"]) # l1 x 4096
        return sents

    def encode_op_post(self, op_node):
        """Encodes OP's post.

        Args:
            op_node: Node of OP's initial post
        Returns:
            op_embs: batch x l1 x in_emb_dim (l1 is num of sentences)
            attn_weights: batch x l1 x 1
        """
        l1 = op_node["sentence_emb"].shape[0]
        if "max" in self.hidden_comb or "mean" in self.hidden_comb:
            op_sents = self.get_sent_embs(op_node) # l1 x in_dim
            op_sents = self.sent_dropout(op_sents)

            op_embs, op_emb_last = \
                    self.sent_encoder(op_sents.unsqueeze(1), 
                                      self.init_sent_hidden()) # l1 x 1 x e
            op_embs = op_embs.permute(1, 0, 2) # 1 x l1 x e

            # Attention weights
            if self.use_attention:
                attn_out = self.attn_layer1(op_embs) # 1 x l1 x 1
                attn_weights = F.softmax(attn_out, dim=1) # 1 x l1 x 1
            else: # Uniform "attention"
                attn_weights = float_var([1 / l1]).repeat(1, l1, 1) # 1 x l1 x 1

        else:
            op_embs = None
            attn_weights = float_var([0]).repeat(1, l1, 1) # 1 x l1 x 1

        return op_embs, attn_weights

    def forward(self, cur_node, op_embs, attn_weights):
        """Encodes the given comment and predicts delta.

        Args:
            cur_node: Node of the current comment
            op_embs: batch x l1 x in_emb_dim
            attn_weights: batch x l1 x 1
        Returns:
            pred_delta: p(delta=1) (batch x 1 x 1)
            inter_emb.data: l1 x l2
            attn_inter_emb.data: l1 x l2
        """
        hidden_in = []  # Input to hidden layer
        cur_sents = self.get_sent_embs(cur_node) # l2 x in_dim
        cur_sents = self.sent_dropout(cur_sents)

        cur_embs, cur_emb_last = \
                self.sent_encoder(cur_sents.unsqueeze(1),
                                  self.init_sent_hidden()) # l2 x b x e
        cur_embs = cur_embs.permute(1, 0, 2) # b x l2 x e
        if "cur" in self.hidden_comb: 
            hidden_in.append(cur_emb_last.permute(1, 0, 2)) # b x 1 x e

        b, l2, _ = cur_embs.size()
        # Compute inter_emb and attn_inter_emb
        if "max" in self.hidden_comb or "mean" in self.hidden_comb:
            _, l1, _ = op_embs.size() # b (batch size) = 1
            if self.interaction == "prod":
                inter_emb = torch.bmm(
                        cur_embs, op_embs.permute(0, 2, 1)) # b x l2 x l1
                inter_emb = inter_emb.view(b, -1, 1) # b x l1*l2 x 1
            else:
                op_embs_rep = op_embs.repeat(1, l2, 1) # b x (l1*l2) x e
                cur_embs_rep = cur_embs.repeat(1, 1, l1)\
                                       .view(b, l1*l2, -1) # b x (l1*l2) x e
                inter_emb = self.inter_layer(torch.cat(
                    (op_embs_rep, cur_embs_rep), dim=2)) # b x (l1*l2) x e
            attn_inter_emb = inter_emb.mul(
                    attn_weights.repeat(1, l2, 1)) # b x (l1*l2) x e

            if "max" in self.hidden_comb:
                max_vec = attn_inter_emb.view(b, l2, -1) # b x l2 x l1*int_e
                if max_vec.size()[1] != 1: # Important due to a PyTorch bug
                    max_vec = max_vec.max(dim=1)[0] # b x l1*int_e
                max_vec = max_vec.view(b, l1, -1) # b x l1 x int_e
                max_vec = max_vec.sum(dim=1) # b x int_e
                max_vec = max_vec.unsqueeze(1) # b x 1 x int_e
                hidden_in.append(max_vec)
            if "mean" in self.hidden_comb:
                hidden_in.append(attn_inter_emb.mean(dim=1).unsqueeze(1))

            # Return values
            inter_emb_data = inter_emb.data[0].view(l2, l1, -1)
            attn_inter_emb_data = attn_inter_emb.data[0].view(l2, l1, -1)

        else:
            inter_emb_data = torch.zeros(l2, 1, 1)
            attn_inter_emb_data = torch.zeros(l2, 1, 1)

        # Tfidf
        if self.tfidf_layer is not None:
            tfidf_in = float_var(cur_node["node_tfidf"].toarray()) # 1 x V
            tfidf_out = self.tfidf_layer(tfidf_in)\
                            .unsqueeze(0) # 1 x 1 x tfidf_emb_dim
            if self.tfidf_type == "B": hidden_in.append(tfidf_out)

        if self.wdsim_type is not None:
            wdsim_in = float_var(cur_node["tfidf_interact"].reshape((1, 1, -1))) # 1 x 1 x 4
            if self.wdsim_type == "B": hidden_in.append(wdsim_in)

        # Feed to hidden layer
        hidden_in = torch.cat(hidden_in, dim=2) # b x 1 x hidden_in_dim
        hidden_out = self.hidden_layer(hidden_in) # b x 1 x hidden_dim
        pred_in = [hidden_out]
        if self.tfidf_type == "A": pred_in.append(tfidf_out)
        if self.wdsim_type == "A": pred_in.append(wdsim_in)
        pred_in = torch.cat(tuple(pred_in), dim=2)
        pred_out = self.predict_layer(pred_in) # b x 1 x 1

        return pred_out, inter_emb_data, attn_inter_emb_data

    def init_sent_hidden(self):
        """Initial hidden state of sentence encoder."""
        return zero_var(1, 1, self.in_emb_dim)

