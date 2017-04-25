import itertools

from models.model import Model, BinaryClassificationMetrics
from models.rnn_classifier import RnnClassifier


comment_sizes = [80, 100, 120, 140]
n_hidden_layers = [80, 100, 120]
vocab_size = [100000, 120000, 140000, 160000]
embedding_size = [32, 64]
input_keep_prob = [0.3, 0.5, 0.7, 1]
output_keep_prob = [0.3, 0.5, 0.7, 1]

#for cs, nh, vs, es, ik, ok in itertools.cartesian_product(comment_sizes

learning_rates = [1.0, 0.1, 0.001]
epsilon = 

                       comment_size: int = 100,
                       batch_size: int = 125,
                       epoch_size: int = 10,
                       n_hidden_layers: int = 120,
                       vocab_size: int = 141000,
                       embedding_size: int = 32,
                       n_classes: int = 2,
                       input_keep_prob: float = 1.0,
                       output_keep_prob: float = 1.0,
                       learning_rate: float = 0.001,
                       beta1: float = 0.9,
                       beta2: float = 0.999,
                       epsilon: float = 1e-08,
                       log_dir: str = fmanip.join('logs', 'rnn')) -> None:
