from typing import List, Tuple, Optional, Dict, cast, TypeVar, Generator
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
from utils.unks import truncate_and_pad, to_words, make_vocab_mapping, vectorize_paragraph, Paragraph, WordId, ParagraphVec, Label, VocabMap

#b

print("Done loading imports")

# Labels
IS_ATTACK = 1
IS_OK = 0

def to_word_array(xs: List[str]) -> List[List[str]]:
    words = [to_words(x) for x in xs]
    max_len = max(len(x) for x in words)
    print("max length:", max_len)
    return [truncate_and_pad(x, 100) for x in words] # DEBUGGING

def add_markers(xs: List[Paragraph]):
    return [["$START"] + x + ["$END"] for x in xs]

def pad(paragraph: Paragraph, lengths: List[int], end: int):
    max_len = max(lengths)
    # pad with UNK; value shouldn't matter since it shouldn't be read
    return [x + [0] * (100 - len(x)) if len(x) <= 100 else
            x[:99] + [end] for x in paragraph]
    # return [x + [0] * (max_len - len(x)) for x in paragraph]

T = TypeVar("T")
def chunks(l: List[T], n: int) -> Generator[List[T], None, None]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class RnnDynamicClassifier(Model[str]):
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
                       input_keep_prob: float = 0.8,
                       output_keep_prob: float = 0.8,
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
        self.y_hot = None       # type: Any
        self.input_keep = None  # type: Any
        self.output_keep = None # type: Any
        self.predictor = None   # type: Any
        self.loss = None        # type: Any
        self.optimizer = None   # type: Any
        self.summary = None     # type: Any
        self.output = None      # type: Any
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
        assert self.y_hot is not None
        assert self.input_keep is not None
        assert self.output_keep is not None
        assert self.predictor is not None
        assert self.loss is not None
        assert self.optimizer is not None
        assert self.summary is not None
        assert self.output is not None
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
        tf.add_to_collection('y_hot', self.y_hot)
        tf.add_to_collection('input_keep', self.input_keep)
        tf.add_to_collection('output_keep', self.output_keep)
        tf.add_to_collection('predictor', self.predictor)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('optimizer', self.optimizer)
        tf.add_to_collection('summary', self.summary)
        tf.add_to_collection('output', self.output)
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
        self.y_hot = tf.get_collection('y_hot')[0]
        self.input_keep = tf.get_collection('input_keep')[0]
        self.output_keep = tf.get_collection('output_keep')[0]
        self.predictor = tf.get_collection('predictor')[0]
        self.loss = tf.get_collection('loss')[0]
        self.optimizer = tf.get_collection('optimizer')[0]
        self.summary = tf.get_collection('summary')[0]
        self.output = tf.get_collection('output')[0]
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

            print('output_shape', self.output.shape)

            self.summary = tf.summary.merge_all()
            self.logger = tf.summary.FileWriter(self._get_log_dir(), graph=tf.get_default_graph())
            self.init = tf.global_variables_initializer()

        #self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session = tf.Session(graph = tf.get_default_graph())

    def _build_input(self) -> None:
        with tf.name_scope('inputs'):
            self.x_input = tf.placeholder(
                    tf.int32, 
                    shape=(None, None), # self.comment_size
                    name='x_input')
            self.y_input = tf.placeholder(
                    tf.int32,
                    shape=(None,),
                    name='y_input')
            self.x_lengths = tf.placeholder(
                    tf.int32,
                    shape=(None,),
                    name='x_lengths')
            self.input_keep = tf.placeholder(
                    tf.float32,
                    shape=tuple(),
                    name='input_keep')
            self.output_keep = tf.placeholder(
                    tf.float32,
                    shape=tuple(),
                    name='output_keep')
            self.y_hot = tf.one_hot(
                    self.y_input,
                    depth=self.n_classes,
                    on_value=tf.constant(1.0, dtype=tf.float32),
                    off_value=tf.constant(0.0, dtype=tf.float32),
                    dtype=tf.float32,
                    name='y_hot_encoded')
            print('y_hot_shape', self.y_hot.shape)

    def _build_predictor(self) -> None:
        with tf.name_scope('prediction'):
            # Make embedding vector for words
            # Shape is [?, vocab_size, embedding_size]
            embedding = tf.Variable(
                    tf.random_uniform([self.vocab_size, self.embedding_size], -1.0, 1.0, dtype=tf.float32),
                    dtype=tf.float32,
                    name="embedding")
            word_vectors = tf.nn.embedding_lookup(embedding, self.x_input)

            self.predictor = self._make_bidirectional_rnn(word_vectors)
            self.loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.predictor, 
                        labels=self.y_hot,),
                    name='loss')

            self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon).minimize(self.loss)

            tf.summary.scalar('loss', self.loss)

    def _build_evaluator(self) -> None:
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(
                    tf.argmax(self.predictor, 1),
                    tf.argmax(self.y_hot, 1))
            accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32),
                    name='accuracy')
            self.output = tf.argmax(self.predictor, 1, name='output')
            self.output_prob = tf.nn.softmax(self.predictor, name='output_prob')

            tf.summary.scalar('batch-accuracy', accuracy)

    def _make_bidirectional_rnn(self, word_vectors: Any) -> Any:
        with tf.name_scope('bidirectional_rnn'):
            # Convert shape of [?, comment_size, embedding_size] into
            # a list of [?, embedding_size]
            # x_unstacked = tf.unstack(word_vectors, self.comment_size, 1)
            output_weight = tf.Variable(
                    tf.random_normal([self.n_hidden_layers * 2, self.n_classes], dtype=tf.float32),
                    dtype=tf.float32,
                    name='output_weight')
            output_bias = tf.Variable(
                    tf.random_normal([self.n_classes], dtype=tf.float32),
                    dtype=tf.float32,
                    name='output_bias')


            # Defining the bidirectional rnn
            # layer = x_unstacked
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
                    
                    (fw_outputs, bw_outputs), (fw_state,bw_state) = tf.nn.bidirectional_dynamic_rnn(
                            forwards_cell,
                            backwards_cell,
                            word_vectors,
                            sequence_length=self.x_lengths,
                            parallel_iterations=self.batch_size,
                            dtype=tf.float32,
                            scope='bidirectional_rnn_{}'.format(i))
                    
                    # # Need to connect outputs
                    # outputs = tf.concat(outputs, 2)
                    # last_output = outputs[:,0,:]

                    # # Use the output of the last rnn cell for classification
                    # prediction = tf.matmul(last_output, output_weight) + output_bias
                    

                    # outputs, fw, bw = tf.contrib.rnn.static_bidirectional_rnn(
                    #         # tf.contrib.rnn.MultiRNNCell(forwards_cells),
                    #         # tf.contrib.rnn.MultiRNNCell(backwards_cells),
                    #         forwards_cell,
                    #         backwards_cell,
                    #         layer,
                    #         dtype=tf.float32,
                    #         sequence_length=self.x_lengths,
                    #         scope='bidirectional_rnn_{}'.format(i))
                    # layer = outputs

            # This is an abuse of scope, but whatever.
            
            # Use the output of the last rnn cell for classification
            foo = tf.layers.batch_normalization(tf.concat([fw_state.h,bw_state.h], axis=1))
            # (batch_size, 2*hidden_size)
            prediction = tf.matmul(foo, output_weight) + output_bias
            return prediction

    def train(self, xs: List[str], ys: List[int], **params: Any) -> None:
        '''Trains the model. The expectation is that this method is called
        exactly once.'''
        if len(params) != 0:
            raise Exception("RNN does not take in any extra params to train")

        xs = np.array(xs)
        ys = np.array(ys)
        bin_edges = np.concatenate([
                np.arange(0,200,20),
                np.arange(200,1000,100),
                np.arange(1000,1500,500),
                np.arange(1500,3000,500)])  # (max is 10000)
        words = add_markers(to_words(x) for x in xs)
        lengths = np.fromiter((len(x) for x in words), dtype=np.int, count=len(xs))
        # hack to add padding to vocab
        fake_padding = ['$PADDING'] * 10000
        self.vocab_map = make_vocab_mapping(words+fake_padding, self.vocab_size)
        vectors = np.array([vectorize_paragraph(self.vocab_map, x) for x in words])
        # (can't create object array from iterator, apparently)

        placements = np.digitize(lengths, bin_edges) - 1
        bins = [np.where(placements==i)[0] for i in range(len(bin_edges))]

        # x_data_raw = to_word_array(xs)
        # x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
        # self.vocab_map = make_vocab_mapping(x_data_raw, self.vocab_size)
        # x_final = [vectorize_paragraph(self.vocab_map, para) for para in x_data_raw]

        # n_batches = len(x_final) // self.batch_size

        self._assert_all_setup()

        self.session.run(self.init)
        for i in range(self.epoch_size):
            # indices = list(range(len(x_final)))
            # random.shuffle(indices)
            # x_lengths_new = [x_lengths[i] for i in indices]
            # x_final_new = [x_final[i] for i in indices]
            # ys_new = [ys[i] for i in indices]

            self.train_epoch(i, bins, self.batch_size, lengths, vectors, ys)

    def train_epoch(self, iteration: int,
                          bins: List[List[int]],
                          batch_size: int,
                          lengths: List[int],
                          xs: List[List[int]],
                          ys: List[int]) -> None:
        start = time.time()
        losses = 0.0
        i = 0
        random.shuffle(bins)

        # Train on dataset
        for bin in bins:
            random.shuffle(bin)
            for batch in chunks(bin, batch_size):
                if i % 10 == 0:
                    print("Training batch", i, end="\r")
                
                batch_data = {
                        self.x_lengths: np.minimum(lengths[batch], 100),
                        self.x_input: pad(xs[batch], lengths[batch], self.vocab_map["$END"]),
                        self.y_input: ys[batch],
                        self.input_keep: self.input_keep_prob,
                        self.output_keep: self.output_keep_prob,
                }

                summary_data, batch_loss, _ = self.session.run(
                        [self.summary, self.loss, self.optimizer], 
                        feed_dict=batch_data)
                losses += batch_loss
                self.logger.add_summary(summary_data, iteration + i)
                i += 1

        # Report results, using last x_batch and y_batch
        delta = time.time() - start 
        print("Iteration {}, avg batch loss = {:.6f}, num batches = {}, time elapsed = {:.3f}".format(
            iteration,
            losses / i,
            i,
            delta))

    def predict(self, xs: List[str]) -> List[List[float]]:
        assert self.vocab_map is not None
        x_data_raw = to_word_array(xs)
        x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
        x_final = [vectorize_paragraph(self.vocab_map, para) for para in x_data_raw]
        batch_data = {
                self.x_input: x_final, 
                self.x_lengths: x_lengths,
                self.input_keep: 1.0,
                self.output_keep: 1.0,
        }
        return cast(List[List[float]], self.session.run(self.output_prob, feed_dict=batch_data))

