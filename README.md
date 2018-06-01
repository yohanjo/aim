# Attentive Interaction Model
 
Jo, Y., Poddar, S., Jeon, B., Shen, Q., Rosé, C. P., & Neubig, G. (2018). Attentive Interaction Model: Modeling Changes in View in Argumentation (pp. 103–116). Presented at the Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), New Orleans, Louisiana: Association for Computational Linguistics.


## Run

The main file is `aim_pair.py`.

Refer to `run_inter.sh` for examples. See the next section for the complete list of input arguments. 


## Command line arguments

```
usage: aim_pair.py [-h] [-domains DOMAINS [DOMAINS ...]]
                   [-domains_cd DOMAINS_CD [DOMAINS_CD ...]]
                   [-vocab_size VOCAB_SIZE] [-max_seq_len MAX_SEQ_LEN]
                   [-max_n_seqs MAX_N_SEQS] [-n_epochs N_EPOCHS] [-attention]
                   [-sent_emb SENT_EMB] [-in_emb_dim IN_EMB_DIM]
                   [-dropout DROPOUT] [-interaction INTERACTION]
                   [-hidden_comb HIDDEN_COMB [HIDDEN_COMB ...]]
                   [-hidden_dim HIDDEN_DIM]
                   [-rank_loss_margin RANK_LOSS_MARGIN]
                   [-rank_loss_weight RANK_LOSS_WEIGHT]
                   [-learning_rate LEARNING_RATE] [-optimizer OPTIMIZER]
                   [-batch_size BATCH_SIZE]

optional arguments:
  -h, --help            show this help message and exit
  -domains DOMAINS [DOMAINS ...]
                        Domains of data to load. (domains7: the in-domains
                        used in the paper) (default: None)
  -domains_cd DOMAINS_CD [DOMAINS_CD ...]
                        Domains for cross-domain evaluationtion. (domains_cd:
                        the cross-domains used in the paper) (default: None)
  -vocab_size VOCAB_SIZE
                        Maximum vocabulary size. (default: 40000)
  -max_seq_len MAX_SEQ_LEN
                        Sequences longer than max length are trimmed.
                        (default: 9223372036854775807)
  -max_n_seqs MAX_N_SEQS
                        Max number of sequences to include in the data.
                        (default: 9223372036854775807)
  -n_epochs N_EPOCHS    Number of epochs. If auto, max is set to 15. (default:
                        auto)
  -attention            Use attention? (default: False)
  -sent_emb SENT_EMB    Sentence embedding method. {infersent, glove}
                        (default: infersent)
  -in_emb_dim IN_EMB_DIM
                        Output dim of sentence encoder. (default: 64)
  -dropout DROPOUT      Dropout ratio. (default: 0.5)
  -interaction INTERACTION
                        Interaction embedding method. (prod: inner product,
                        m1_..._mL: feed-forward network with mi being the num
                        of nodes on the i-th layer) (default: prod)
  -hidden_comb HIDDEN_COMB [HIDDEN_COMB ...]
                        Prediction input. (max: MAX, cur: HSENT, wsdim: WDO,
                        tfidf.A.n: TFIDF with a feedforward net with n output
                        nodes) (default: ['cur', 'max'])
  -hidden_dim HIDDEN_DIM
                        Dim of the last layer before classifier. (default: 32)
  -rank_loss_margin RANK_LOSS_MARGIN
                        Margin for the margin rank loss. (default: 0.5)
  -rank_loss_weight RANK_LOSS_WEIGHT
                        Ranking loss weight. (default: 1)
  -learning_rate LEARNING_RATE
                        Learning rate. If 'auto', set to 1e-4 for Adam, 1e-2
                        for SGD (default: auto)
  -optimizer OPTIMIZER  Optimizer {adam, adamax, sgd} (default: adamax)
  -batch_size BATCH_SIZE
                        Minibatch size. (default: 1)
```

## Data format

To run the model, `data` directory must be located in the root directory. Under `data`, the following files must exist:
 * `voca_40000.txt`: Vocabulary text file, with each line being a word.
 * `V40000_[DOMAIN]_[SPLIT].p`: Pickled input data file (details below). Domains are specified in `constants.py`. A split is either `train`, `val`, or `test`. 

Each pickle file is a json list (array) of tuples (`op_node`, `comments`). `op_node` is the OH's post of a discussion, and `comments` is the list of `comment`s that have a delta label. `op_node` and `comment` have the following attributes:
 * `sentence_emb`: Input sentence embedding pre-trained by Conneau et al.'s model.
 * `node_tfidf` (only for `comment`): TFIDF vector.
 * `tfidf_interact` (only for `comment`): Word overlap vector.
 * `delta` (only for `comment`): True delta (1 or 0). 
