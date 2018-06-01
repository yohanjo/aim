MODEL_DIR = "../models"
DATA_DIR = "../data"
LOG_DIR = "../logs"
PLOT_DIR = "../logs"
FEAT_DIR = "../features"
VOCA_PATH = "{}/voca_40000.txt".format(DATA_DIR)
SPLIT_PATH = "{}/splits.csv".format(DATA_DIR)
GLOVE_PATH = "{}/glove.840B.300d.txt".format(MODEL_DIR)
GLOVE_NUMPY_PATH = "{}/glove.numpy.p".format(MODEL_DIR)
GLOVE_DIM = 300
SENT_EMB_PATH = "{}/sent_emb/sent_emb.p".format(FEAT_DIR)
SENTINFER_MODEL_PATH = "{}/sent_emb/infersent.allnli.pickle".format(FEAT_DIR)

DOMAINS2 = ["education", "food"]
DOMAINS7 = ["clothing", "food", "education", "art", "driving", 
            "computer", "life"]
DOMAINS_CD = ["government", "money", "tags", "animal",
               "commonwords", "drug", "gender", "law", "reddit", "weapon",
               "god", "sports", "world"]
DOMAINS_ALL = ["clothing", "food", "education", "art", "driving", 
               "computer", "life", "government", "money", "tags", "animal",
               "commonwords", "drug", "gender", "law", "reddit", "weapon",
               "god", "sports", "world"]


VOCAB_SIZE = 40000
N_TOPICS = 100
SENT_EMB_DIM = 4096
TRANSACT_DIM = 1

TOPIC_DIST_PATH = "{}/topic/V40000-T100-NodeTopicDist.csv".format(FEAT_DIR)

UNK_IDX = 0
SOS_IDX = 1
EOS_IDX = 2
