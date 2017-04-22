from typing import List, Tuple, Optional, Dict 
import os.path
import shutil
from collections import Counter

import numpy as np
import tensorflow as tf
import nltk
import sklearn.metrics

import utils.file_manip as fmanip
from parsing import load_raw_data
from custom_types import *

from models.model import Model, ClassificationMetrics


print("Done loading imports")

# Labels
IS_ATTACK = 1
IS_OK = 0

# Type aliases, for readability
Paragraph = List[str]
WordId = int
ParagraphVec = List[WordId]
Label = int

def truncate_and_pad(paragraph: Paragraph, max_length: int) -> Paragraph:
    # Subtract 2 so we have space for the start and end tokens
    length = min(len(paragraph), max_length - 2)
    if length < max_length - 2:
        padding = max_length - 2 - length
    else:
        padding = 0

    out = ["$START"] + paragraph[:length] + ["$END"] + (["$PADDING"] * padding)
    return out

def make_vocab_mapping(x: List[Paragraph],
                       max_vocab_size: Optional[int] = None) -> Dict[str, WordId]:
    freqs = Counter()  # type: Counter[str]
    for paragraph in x:
        for word in paragraph:
            freqs[word] = freqs.get(word, 0)
    out = {'$UNK': 0}
    count = 1
    if max_vocab_size is None:
        max_vocab_size = len(freqs)
    print("Actual vocab size: {}".format(len(freqs)))
    for key, num in freqs.most_common(max_vocab_size - 1):
        if num == 1:
            continue

        out[key] = count
        count += 1
    return out

def vectorize_paragraph(vocab_map: Dict[str, WordId], para: Paragraph) -> List[WordId]:
    unk_id = vocab_map['$UNK']
    return [vocab_map.get(word, unk_id) for word in para]

class RnnClassifier(Model[str]):
    def __init__(self, paragraph_size: int = 100,
                       batch_size: int = 120,
                       epoch_size: int = 10,
                       n_hidden_layers: int = 120,
                       vocab_size: int = 141000,
                       embedding_size: int = 32,
                       n_classes: int = 2,
                       log_dir: str = fmanip.join('logs', 'rnn')) -> None:
        '''For the sake of consistency, the constructor for all subclasses
        should take in an arbitrary set of parameters, and do very little
        else. All parameters should have a default value.'''
        # Hyperparameters
        self.paragraph_size = paragraph_size
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.n_hidden_layers = n_hidden_layers
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.log_dir = log_dir

        # Particular tensorflow nodes worth keeping a reference to
        # Types are set to Any because mypy doesn't yet understand
        # the tensorflow library
        self.x_input = None     # type: Any
        self.y_input = None     # type: Any
        self.predictor = None   # type: Any
        self.loss = None        # type: Any
        self.optimizer = None   # type: Any
        self.summary = None     # type: Any
        self.output = None      # type: Any
        self.init = None        # type: Any
        self.logger = None      # type: Any
        self.session = None     # type: Any

        self.vocab_map = None   # type: Optional[Dict[str, WordId]]

    def _assert_all_setup(self) -> None:
        assert self.x_input is not None
        assert self.y_input is not None
        assert self.predictor is not None
        assert self.loss is not None
        assert self.optimizer is not None
        assert self.summary is not None
        assert self.output is not None
        assert self.init is not None
        assert self.logger is not None
        assert self.session is not None
        assert self.vocab_map is not None

    def get_parameters(self) -> Dict[str, Any]:
        return {
                'paragraph_size': self.paragraph_size,
                'batch_size': self.batch_size,
                'epoch_size': self.epoch_size,
                'n_hidden_layers': self.n_hidden_layers,
                'embedding_size': self.embedding_size,
                'n_classes': self.n_classes,
                'log_dir': self.log_dir,
        }

    def _save_model(self, path: str) -> None:
        with open(fmanip.join(path, 'vocab_map.json'), 'w') as stream:
            json.dump(self.vocab_map, stream)
        tf.train.export_meta_graph(filename=fmanip.join(path, 'tensorflow_graph.meta'))

    def _restore_model(self, path: str) -> None:
        raise NotImplementedError()

    def build_model(self) -> None:
        '''Builds and saves the model, using the currently set params.'''
        with tf.name_scope('rnn-classifier'):
            self._build_input()
            self._build_predictor()
            self._build_evaluator()

            self.summary = tf.summary.merge_all()
            self.logger = tf.summary.FileWriter(self.log_dir, graph=tf.get_default_graph())
            self.init = tf.global_variables_initializer()

        #self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session = tf.Session()

    def _build_input(self) -> None:
        with tf.name_scope('inputs'):
            self.x_input = tf.placeholder(
                    tf.int32, 
                    shape=(None, self.paragraph_size),
                    name='x_input')
            self.y_input = tf.placeholder(
                    tf.float32,
                    shape=(None, self.n_classes),
                    name='y_input')

    def _build_predictor(self) -> None:
        with tf.name_scope('prediction'):
            # Make embedding vector for words
            # Shape is [?, vocab_size, embedding_size]
            embedding = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0),
                    name="embedding")
            word_vectors = tf.nn.embedding_lookup(embedding, self.x_input)

            self.predictor = self._make_bidirectional_rnn(word_vectors)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.predictor, 
                    labels=self.y_input),
                    name='loss')
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

            tf.summary.scalar('loss', self.loss)

    def _build_evaluator(self) -> None:
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(
                    tf.argmax(self.predictor, 1), 
                    tf.argmax(self.y_input, 1))
            accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32),
                    name='accuracy')
            self.output = tf.argmax(self.predictor, 1)

            tf.summary.scalar('batch-accuracy', accuracy)

    def _make_bidirectional_rnn(self, word_vectors):
        with tf.name_scope('bidirectional_rnn'):
            # Convert shape of [?, paragraph_size, embedding_size] into
            # a list of [?, embedding_size]
            x_unstacked = tf.unstack(word_vectors, self.paragraph_size, 1)

            output_weight = tf.Variable(
                    tf.random_normal([2 * self.n_hidden_layers, self.n_classes]),
                    name='output_weight')
            output_bias = tf.Variable(
                    tf.random_normal([self.n_classes]),
                    name='output_bias')

            # Defining the bidirectional rnn
            forwards_lstm = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_layers)
            backwards_lstm = tf.contrib.rnn.BasicLSTMCell(self.n_hidden_layers)

            outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                    forwards_lstm,
                    backwards_lstm,
                    x_unstacked,
                    dtype=tf.float32)

            # Use the output of the last rnn cell for classification
            prediction = tf.matmul(outputs[-1], output_weight) + output_bias
            return prediction

    def train(self, xs: List[str], ys: List[List[bool]]) -> None:
        '''Trains the model. The expectation is that this method is called
        exactly once.'''
        x_data_raw = [truncate_and_pad(nltk.word_tokenize(x), self.paragraph_size) for x in xs]
        self.vocab_map = make_vocab_mapping(x_data_raw, self.vocab_size)
        x_final = [vectorize_paragraph(self.vocab_map, para) for para in x_data_raw]

        n_batches = len(x_final) // self.batch_size

        self._assert_all_setup()

        self.session.run(self.init)
        for i in range(self.epoch_size):
            self.train_epoch(i, n_batches, x_final, ys)

    def train_epoch(self, iteration: int,
                          n_batches: int, 
                          xs: List[List[int]], 
                          ys: List[List[bool]]) -> None:
        # Train on dataset
        for batch_num in range(n_batches):
            start_idx = batch_num * self.batch_size
            end_idx = (batch_num + 1) * self.batch_size

            x_batch = xs[start_idx: end_idx]
            y_batch = ys[start_idx: end_idx]

            batch_data = {self.x_input: x_batch, self.y_input: y_batch}

            self.session.run(self.optimizer, feed_dict=batch_data)

        # Report results, using last x_batch and y_batch
        summary_data, batch_loss = self.session.run(
                [self.summary, self.loss], 
                feed_dict=batch_data)

        self.logger.add_summary(summary_data, iteration)
        print("Iteration {}, batch loss = {:.6f}".format(iteration, batch_loss))

    def predict(self, xs: List[str]) -> List[List[bool]]:
        x_data_raw = [truncate_and_pad(nltk.word_tokenize(x), self.paragraph_size) for x in xs]
        x_final = [vectorize_paragraph(self.vocab_map, para) for para in x_data_raw]
        batch_data = {self.x_input: x_final}
        return self.session.run(self.output, feed_dict=batch_data)


def make_bidirectional_rnn(x_data, n_words, n_hidden_layers, n_classes):
    with tf.name_scope('bidirectional_rnn'):
        # Convert shape of [?, paragraph_size, embedding_size] into
        # a list of [?, embedding_size]
        x_unstacked = tf.unstack(x_data, n_words, 1)

        output_weight = tf.Variable(
                tf.random_normal([2 * n_hidden_layers, n_classes]),
                name='output_weight')
        output_bias = tf.Variable(
                tf.random_normal([n_classes]),
                name='output_bias')

        # Defining the bidirectional rnn
        forwards_lstm = tf.contrib.rnn.BasicLSTMCell(n_hidden_layers)
        backwards_lstm = tf.contrib.rnn.BasicLSTMCell(n_hidden_layers)

        outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                forwards_lstm,
                backwards_lstm,
                x_unstacked,
                dtype=tf.float32)

        # Use the output of the last rnn cell for classification
        prediction = tf.matmul(outputs[-1], output_weight) + output_bias
        return prediction


def extract_data(comments: List[Comment]) -> Tuple[List[str], List[List[bool]]]:
    x_values = []
    y_values = []
    for comment in comments:
        x_values.append(comment.comment)
        cls = IS_ATTACK if comment.average.attack > 0.5 else IS_OK
        y_values.append([cls == IS_OK, cls == IS_ATTACK])
    return x_values, y_values


def main2() -> None:
    print("Starting...")

    # Meta parameters
    data_src_path = 'data/wikipedia-detox-data-v6'

    # Classifier setup
    print("Building model...")
    classifier = RnnClassifier(n_classes=3)
    classifier.build_model()
    shutil.rmtree(classifier.log_dir, ignore_errors=True)

    # Load data
    print("Loading data...")
    train_data, dev_data, test_data = load_raw_data(data_src_path)
    x_train_raw, y_train = extract_data(train_data)
    x_dev_raw, y_dev = extract_data(dev_data)

    # Begin training
    print("Training...")
    classifier.train(x_train_raw, y_train)

    # Evaluation
    print("Evaluation...")
    y_true = np.argmax(np.asarray(y_dev), 1)
    y_predicted = classifier.predict(x_dev_raw)
    metrics = ClassificationMetrics(y_true, y_predicted)

    print(metrics.get_header())
    print(metrics.to_table_row())
    print(metrics.confusion_matrix)

if __name__ == '__main__':
    main2()


