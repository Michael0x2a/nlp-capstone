from typing import List, Tuple, Optional, Dict, cast
import os.path
import shutil
import json
import time  # type: ignore
import random
from collections import Counter
import re

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import nltk  # type: ignore
import nltk.corpus  # type: ignore
import sklearn.metrics  # type: ignore

import utils.file_manip as fmanip
from data_extraction.wikipedia import * 
from custom_types import *

from models.model import Model


print("Done loading imports")

# Labels
IS_ATTACK = 1
IS_OK = 0

# Type aliases, for readability
Paragraph = List[str]
WordId = int
ParagraphVec = List[WordId]
Label = int

stop_words = set(nltk.corpus.stopwords.words('english'))

# Copied from https://mathiasbynens.be/demo/url-regex
# using @imme_emosol's implementation
_url = re.compile('(https?|ftp)://(-\\.)?([^\\s/?\\.#-]+\\.?)+(/[^\\s]*)?')
_missing_space_after_period = re.compile('([a-z])\\.([a-zA-Z])')
_quotes_at_start = re.compile("''([a-zA-Z])")
_colons_at_start = re.compile('\\:+([a-zA-Z0-9])')

def to_words(inputs: str) -> List[str]:
    #inputs = re.sub(_url, '$UNK-URL', inputs)
    inputs = inputs.replace('=====', '')
    inputs = inputs.replace('====', '')
    inputs = inputs.replace('===', '')
    inputs = inputs.replace('==', '')
    inputs = inputs.replace('`', "'")
    inputs = re.sub(_quotes_at_start, (lambda match: "'' {}".format(match.group(1))), inputs)
    inputs = re.sub(_colons_at_start, (lambda match: match.group(1)), inputs)
    words = nltk.word_tokenize(inputs)
    out = []
    for word in words:
        if word.startswith('//') and word != '//':
            out.append('$UNK-URL')
        else:
            out.append(word)
    
    return out


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
    words_set = set()
    for paragraph in x:
        for word in paragraph:
            freqs[word] = freqs.get(word, 0) + 1
            words_set.add(word)
    out = {'$UNK': 0}
    count = 1
    if max_vocab_size is None:
        max_vocab_size = len(freqs)
    print("Actual vocab size: {}".format(len(freqs)))
    for key, num in freqs.most_common(max_vocab_size - 1):
        #if num == 1:
        #    continue
        out[key] = count
        count += 1
    oov_count = 0
    with open("unks.txt", "w") as stream:
        for word in words_set:
            if word not in out:
                stream.write(word)
                stream.write("\n")
                oov_count += freqs[word]
    total = sum(freqs.values())
    print("OOV count: {}/{} ({})".format(oov_count, total, oov_count/total))
    return out

def vectorize_paragraph(vocab_map: Dict[str, WordId], para: Paragraph) -> List[WordId]:
    unk_id = vocab_map['$UNK']
    return [vocab_map.get(word, unk_id) for word in para]

class RnnClassifierSoft(Model[str]):
    base_log_dir = "runs/rnn/run{}"

    '''
    RNN classifier

    --comment_size [int; default=100]
        How long to cap the length of each comment (padding if
        the comment is shorter)

    --batch_size [int; default=125]

    --epoch_size [int; default=10]

    --n_hidden_layers [int; default=120]

    --vocab_size [int; default=141000]

    --embedding_size [int; default=32]
    '''
    def __init__(self, restore_from: Optional[str] = None,
                       run_num: Optional[int] = None,
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
                       epsilon: float = 1e-08) -> None:

        # Hyperparameters
        self.comment_size = comment_size
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        self.n_hidden_layers = n_hidden_layers
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.n_classes = n_classes
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        # Particular tensorflow nodes worth keeping a reference to
        # Types are set to Any because mypy doesn't yet understand
        # the tensorflow library
        self.x_input = None     # type: Any
        self.y_input = None     # type: Any
        self.x_lengths = None   # type: Any
        self.input_keep = None  # type: Any
        self.output_keep = None # type: Any
        self.predictor = None   # type: Any
        self.loss = None        # type: Any
        self.optimizer = None   # type: Any
        self.summary = None     # type: Any
        self.output_prob = None # type: Any
        self.init = None        # type: Any
        self.logger = None      # type: Any
        self.session = None     # type: Any

        self.vocab_map = None   # type: Optional[Dict[str, WordId]]

        super().__init__(restore_from, run_num)

        if restore_from is None:
            self._build_model()

    def _assert_all_setup(self) -> None:
        assert self.x_input is not None
        assert self.y_input is not None
        assert self.x_lengths is not None
        assert self.input_keep is not None
        assert self.output_keep is not None
        assert self.predictor is not None
        assert self.loss is not None
        assert self.optimizer is not None
        assert self.summary is not None
        assert self.output_prob is not None
        assert self.init is not None
        assert self.logger is not None
        assert self.session is not None
        assert self.vocab_map is not None

    def _get_parameters(self) -> Dict[str, Any]:
        return {
                'comment_size': self.comment_size,
                'batch_size': self.batch_size,
                'epoch_size': self.epoch_size,
                'n_hidden_layers': self.n_hidden_layers,
                'embedding_size': self.embedding_size,
                'n_classes': self.n_classes,
                'input_keep_prob': self.input_keep_prob,
                'output_keep_prob': self.output_keep_prob,
                'learning_rate': self.learning_rate,
                'beta1': self.beta1,
                'beta2': self.beta2,
                'epsilon': self.epsilon,
        }

    def _save_model(self, path: str) -> None:
        with open(fmanip.join(path, 'vocab_map.json'), 'w') as stream:
            json.dump(self.vocab_map, stream)
        saver = tf.train.Saver()

        tf.add_to_collection('x_input', self.x_input)
        tf.add_to_collection('y_input', self.y_input)
        tf.add_to_collection('x_lengths', self.x_lengths)
        tf.add_to_collection('input_keep', self.input_keep)
        tf.add_to_collection('output_keep', self.output_keep)
        tf.add_to_collection('predictor', self.predictor)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('optimizer', self.optimizer)
        tf.add_to_collection('summary', self.summary)
        # tf.add_to_collection('output', self.output)
        tf.add_to_collection('output_prob', self.output_prob)
        tf.add_to_collection('init', self.init)

        saver.save(self.session, fmanip.join(path, 'model'))
        tf.train.export_meta_graph(filename=fmanip.join(path, 'tensorflow_graph.meta'))

    def _restore_model(self, path: str) -> None:
        with open(fmanip.join(path, 'vocab_map.json'), 'r') as stream:
            self.vocab_map = json.load(stream)

        self.session = tf.Session(graph = tf.get_default_graph())
        saver = tf.train.import_meta_graph(fmanip.join(path, 'tensorflow_graph.meta'))
        saver.restore(self.session, fmanip.join(path, 'model'))

        self.x_input = tf.get_collection('x_input')[0]
        self.y_input = tf.get_collection('y_input')[0]
        self.x_lengths = tf.get_collection('x_lengths')[0]
        self.input_keep = tf.get_collection('input_keep')[0]
        self.output_keep = tf.get_collection('output_keep')[0]
        self.predictor = tf.get_collection('predictor')[0]
        self.loss = tf.get_collection('loss')[0]
        self.optimizer = tf.get_collection('optimizer')[0]
        self.summary = tf.get_collection('summary')[0]
        self.output_prob = tf.get_collection('output_prob')[0]
        self.init = tf.get_collection('init')[0]
        self.logger = tf.summary.FileWriter(self._get_log_dir(), graph=tf.get_default_graph())

        self._assert_all_setup()

    def _build_model(self) -> None:
        '''Builds the model, using the currently set params.'''
        with tf.name_scope('rnn-classifier'):
            self._build_input()
            self._build_predictor()
            self._build_evaluator()

            self.summary = tf.summary.merge_all()
            self.logger = tf.summary.FileWriter(self._get_log_dir(), graph=tf.get_default_graph())
            self.init = tf.global_variables_initializer()

        #self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session = tf.Session(graph = tf.get_default_graph())

    def _build_input(self) -> None:
        with tf.name_scope('inputs'):
            self.x_input = tf.placeholder(
                    tf.int64, 
                    shape=(None, self.comment_size),
                    name='x_input')
            self.y_input = tf.placeholder(
                    tf.float64,
                    shape=(None,),
                    name='y_input')
            self.x_lengths = tf.placeholder(
                    tf.int64,
                    shape=(None,),
                    name='x_lengths')
            self.input_keep = tf.placeholder(
                    tf.float64,
                    shape=tuple(),
                    name='input_keep')
            self.output_keep = tf.placeholder(
                    tf.float64,
                    shape=tuple(),
                    name='output_keep')

    def _build_predictor(self) -> None:
        with tf.name_scope('prediction'):
            # Make embedding vector for words
            # Shape is [?, vocab_size, embedding_size]
            embedding = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0, dtype=tf.float64),
                    dtype=tf.float64,
                    name="embedding")
            word_vectors = tf.nn.embedding_lookup(embedding, self.x_input)

            labels = tf.cast(
                    tf.map_fn(
                        lambda x: x > 0.0, 
                        self.y_input,
                        dtype=tf.bool),
                    tf.int64)
            self.predictor = self._make_bidirectional_rnn(word_vectors)
            self.loss = tf.losses.hinge_loss(
                    labels=labels, 
                    logits=self.predictor)


            self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon).minimize(self.loss)

            tf.summary.scalar('loss', self.loss)

    def _build_evaluator(self) -> None:
        with tf.name_scope('evaluation'):
            self.output_prob = self.predictor  #tf.nn.softmax(self.predictor, name='output_prob')

    def _make_bidirectional_rnn(self, word_vectors: Any) -> Any:
        with tf.name_scope('bidirectional_rnn'):
            # Convert shape of [?, comment_size, embedding_size] into
            # a list of [?, embedding_size]
            x_unstacked = tf.unstack(word_vectors, self.comment_size, 1)
            output_weight = tf.Variable(
                    tf.random_normal([self.n_hidden_layers * 2, 1], dtype=tf.float64),
                    dtype=tf.float64,
                    name='output_weight')
            output_bias = tf.Variable(
                    tf.random_normal([1], dtype=tf.float64),
                    dtype=tf.float64,
                    name='output_bias')


            # Defining the bidirectional rnn
            layer = x_unstacked
            for i in range(1):
                with tf.name_scope('layer_{}'.format(i)):
                    forwards_cell = tf.contrib.rnn.DropoutWrapper(
                            tf.contrib.rnn.BasicLSTMCell(self.n_hidden_layers),
                            input_keep_prob=self.input_keep,
                            output_keep_prob=self.output_keep)
                    backwards_cell = tf.contrib.rnn.DropoutWrapper(
                            tf.contrib.rnn.BasicLSTMCell(self.n_hidden_layers),
                            input_keep_prob=self.input_keep,
                            output_keep_prob=self.output_keep)
                    #forwards_cell = tf.contrib.rnn.GRUCell(self.n_hidden_layers)
                    #backwards_cell = tf.contrib.rnn.GRUCell(self.n_hidden_layers)

                    forwards_cells = [tf.contrib.rnn.DropoutWrapper(
                            tf.contrib.rnn.BasicLSTMCell(self.n_hidden_layers),
                            input_keep_prob=self.input_keep,
                            output_keep_prob=self.output_keep) for i in range(2)]
                    backwards_cells = [tf.contrib.rnn.DropoutWrapper(
                            tf.contrib.rnn.BasicLSTMCell(self.n_hidden_layers),
                            input_keep_prob=self.input_keep,
                            output_keep_prob=self.output_keep) for i in range(2)]
                    '''
                    outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                            forwards_cell,
                            backwards_cell,
                            #x_unstacked,
                            inputs=word_vectors,
                            sequence_length=self.x_lengths,
                            dtype=tf.float64)
                    
                    # Need to connect outputs
                    outputs = tf.concat(outputs, 2)
                    last_output = outputs[:,0,:]

                    # Use the output of the last rnn cell for classification
                    prediction = tf.matmul(last_output, output_weight) + output_bias
                    '''

                    outputs, _, _ = tf.contrib.rnn.static_bidirectional_rnn(
                            tf.contrib.rnn.MultiRNNCell(forwards_cells),
                            tf.contrib.rnn.MultiRNNCell(backwards_cells),
                            #forwards_cell,
                            #backwards_cell,
                            layer,
                            dtype=tf.float64,
                            scope='bidirectional_rnn_{}'.format(i))
                    layer = outputs

            # This is an abuse of scope, but whatever.
            
            # Use the output of the last rnn cell for classification
            #foo = tf.layers.batch_normalization(outputs[-1])
            prediction = tf.matmul(outputs[-1], output_weight) + output_bias
            return tf.squeeze(prediction)

    def train(self, xs: List[str], ys: List[int], **params: Any) -> None:
        '''Trains the model. The expectation is that this method is called
        exactly once.'''
        if len(params) != 0:
            raise Exception("RNN does not take in any extra params to train")

        x_data_raw = [truncate_and_pad(to_words(x), self.comment_size) for x in xs]
        x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
        self.vocab_map = make_vocab_mapping(x_data_raw, self.vocab_size)
        x_final = [vectorize_paragraph(self.vocab_map, para) for para in x_data_raw]

        n_batches = len(x_final) // self.batch_size

        self._assert_all_setup()

        self.session.run(self.init)
        for i in range(self.epoch_size):
            indices = list(range(len(x_final)))
            random.shuffle(indices)
            x_lengths_new = [x_lengths[i] for i in indices]
            x_final_new = [x_final[i] for i in indices]
            ys_new = [ys[i] for i in indices]

            self.train_epoch(i, n_batches, x_lengths, x_final, ys)

    def train_epoch(self, iteration: int,
                          n_batches: int, 
                          x_lengths: List[int],
                          xs: List[List[int]], 
                          ys: List[int]) -> None:
        start = time.time()

        losses = 0.0

        # Train on dataset
        for batch_num in range(n_batches):
            start_idx = batch_num * self.batch_size
            end_idx = (batch_num + 1) * self.batch_size

            x_batch = xs[start_idx: end_idx]
            y_batch = ys[start_idx: end_idx]
            x_len_batch = x_lengths[start_idx: end_idx]

            batch_data = {
                    self.x_lengths: x_len_batch, 
                    self.x_input: x_batch, 
                    self.y_input: y_batch,
                    self.input_keep: self.input_keep_prob,
                    self.output_keep: self.output_keep_prob,
            }

            summary_data, batch_loss, _ = self.session.run(
                    [self.summary, self.loss, self.optimizer], 
                    feed_dict=batch_data)
            losses += batch_loss

        # Report results, using last x_batch and y_batch
        self.logger.add_summary(summary_data, iteration)
        delta = time.time() - start 
        print("Iteration {}, avg batch loss = {:.6f}, num batches = {}, time elapsed = {:.3f}".format(
            iteration, 
            losses / n_batches, 
            n_batches,
            delta))

    def predict(self, xs: List[str]) -> List[List[float]]:
        assert self.vocab_map is not None
        x_data_raw = [truncate_and_pad(to_words(x), self.comment_size) for x in xs]
        x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
        x_final = [vectorize_paragraph(self.vocab_map, para) for para in x_data_raw]
        batch_data = {
                self.x_input: x_final, 
                self.x_lengths: x_lengths,
                self.input_keep: 1.0,
                self.output_keep: 1.0,
        }
        return cast(List[List[float]], self.session.run(self.output_prob, feed_dict=batch_data))

