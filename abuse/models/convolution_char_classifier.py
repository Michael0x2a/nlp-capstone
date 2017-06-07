from typing import Optional, Dict, Any, List, cast
import time
import math

import tensorflow as tf
import numpy as np

from models.model import Model
from utils.unks import prep_train_char, shuffle, prep_test_char, VocabMap
import utils.file_manip as fmanip

class ConvolutionCharClassifier(Model[str]):
    # default log dir; override this
    base_log_dir = "runs/conv_char/run{}"

    def __init__(self,
                 use_padding: bool = True,
                 stride: int = 1,
                 dropout: float = 0.5,
                 num_filters: int = 300,
                 filter_sizes: str = '2,3,4,5,6,7',
                 comment_size: int = 555,
                 epoch_size: int = 15,
                 batch_size: int = 125,
                 vocab_size: int = 100,
                 embedding_size: int = 64,
                 restore_from: Optional[str] = None,
                 run_num: Optional[int]=None
                 ) -> None:
        # Hyperparameters
        self.use_padding = use_padding
        self.stride = stride
        self.dropout_prob = dropout
        self.num_filters = num_filters
        self.filter_sizes = [int(x) for x in filter_sizes.split(',')]
        self.comment_size = comment_size
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size

        # Stored results
        self.vocab_map = None  # type: Optional[VocabMap]

        # Tensorflow nodes
        # Inputs
        self.x_input = None    # type: Any
        self.y_input = None    # type: Any
        self.dropout = None    # type: Any
        self.y_hot = None      # type: Any

        # Metadata
        self.summary = None    # type: Any
        self.logger = None     # type: Any
        self.init = None       # type: Any

        # Training
        self.loss = None       # type: Any
        self.optimizer = None  # type: Any
        self.predictor = None  # type: Any
        self.output = None      # type: Any
        self.output_prob = None  # type: Any

        self.session = None  # type: Any

        super().__init__(restore_from, run_num)
        if restore_from is None:
            self._build_model()

    # Core methods that must be implemented

    def _get_parameters(self) -> Dict[str, Any]:
        return {
                'use_padding': self.use_padding,
                'stride': self.stride,
                'dropout_prob': self.dropout_prob,
                'num_filters': self.num_filters,
                'filter_sizes': ','.join(str(x) for x in self.filter_sizes),
                'comment_size': self.comment_size,
                'epoch_size': self.epoch_size,
                'batch_size': self.batch_size,
                'vocab_size': self.vocab_size,
                'embedding_size': self.embedding_size,
        }

    def _save_model(self, path: str) -> None:
        '''Saves the model. The path is a path to an existing folder;
        this method may create any arbitrary files/folders within the
        provided path.'''
        with open(fmanip.join(path, 'vocab_map.json'), 'w') as stream:
            json.dump(self.vocab_map, stream)
        saver = tf.train.Saver()

        tf.add_to_collection('x_input', self.x_input)
        tf.add_to_collection('dropout', self.dropout)
        tf.add_to_collection('y_input', self.y_input)
        tf.add_to_collection('y_hot', self.y_hot)
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
        self.dropout = tf.get_collection('dropout')[0]
        self.y_input = tf.get_collection('y_input')[0]
        self.predictor = tf.get_collection('predictor')[0]
        self.loss = tf.get_collection('loss')[0]
        self.optimizer = tf.get_collection('optimizer')[0]
        self.summary = tf.get_collection('summary')[0]
        self.output = tf.get_collection('output')[0]
        self.output_prob = tf.get_collection('output_prob')[0]
        self.init = tf.get_collection('init')[0]
        self.logger = tf.summary.FileWriter(self._get_log_dir(), graph=tf.get_default_graph())

    def train(self, xs: List[str], ys: List[int], **params: Any) -> None:
        '''Trains the model. The expectation is that this method is called
        exactly once. The model can also accept additional params to tweak
        the behavior of the training method in some way. Note that cmd.py
        will completely ignore the kwargs, so the 'train' method shouldn't
        rely on any of them being present.'''
        if len(params) != 0:
            raise Exception("RNN does not take in any extra params to train")

        avg = sum(map(len, xs))
        print("Average sentence size:", avg / len(xs))

        x_final, x_lengths, vocab_map = prep_train_char(xs, self.comment_size, self.vocab_size)
        self.vocab_map = vocab_map
        n_batches = len(x_final) // self.batch_size

        self.session.run(self.init)
        for i in range(self.epoch_size):
            x_final_new, _, ys_new = shuffle(x_final, x_lengths, ys)
            self.train_epoch(i, n_batches, x_final, ys)

    def train_epoch(self, iteration: int,
                          n_batches: int, 
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

            batch_data = {
                    self.x_input: x_batch, 
                    self.dropout: self.dropout_prob,
                    self.y_input: y_batch,
            }

            summary_data, batch_loss, _ = self.session.run(
                    [self.summary, self.loss, self.optimizer], 
                    feed_dict=batch_data)
            losses += batch_loss
            self.logger.add_summary(summary_data, batch_num + n_batches * iteration)

        # Report results, using last x_batch and y_batch
        delta = time.time() - start 
        print("Iteration {}, avg batch loss = {:.6f}, num batches = {}, time elapsed = {:.3f}".format(
            iteration, 
            losses / n_batches, 
            n_batches,
            delta))

    def predict(self, xs: List[str]) -> List[List[float]]:
        batches = math.ceil(len(xs) / self.batch_size)
        out = []
        for n in range(batches):
            start = n * self.batch_size
            end = (n + 1) * self.batch_size

            xs_small = xs[start: end]

            x_final, _ = prep_test_char(xs_small, self.comment_size, self.vocab_map)
            batch_data = {
                    self.x_input: x_final, 
                    self.dropout: 1.0,
            }
            out.append(self.session.run(self.output_prob, feed_dict=batch_data))

        return cast(List[List[float]], np.concatenate(out))


    def _build_model(self) -> None:
        with tf.name_scope('convolution-classifier'):
            self._build_input()
            self._build_predictor()
            self._build_evaluator()

            self.summary = tf.summary.merge_all()
            self.logger = tf.summary.FileWriter(self._get_log_dir(), graph=tf.get_default_graph())
            self.init = tf.global_variables_initializer()

        self.session = tf.Session(graph = tf.get_default_graph())

    def _build_input(self) -> None:
        with tf.name_scope('inputs'):
            self.x_input = tf.placeholder(
                    tf.int32,
                    shape=(None, self.comment_size),
                    name='x_input')
            self.dropout = tf.placeholder(
                    tf.float32,
                    shape=tuple(),
                    name='dropout')
            self.y_input = tf.placeholder(
                    tf.int32,
                    shape=(None,),
                    name='y_input')
            self.y_hot = tf.one_hot(
                    self.y_input,
                    depth=2,
                    on_value=tf.constant(1.0, dtype=tf.float32),
                    off_value=tf.constant(0.0, dtype=tf.float32),
                    name='y_hot_encoded')

    def _build_predictor(self) -> None:
        with tf.name_scope('prediction'):
            # Train embedding; TODO: Try using glove or word2vec?
            embedding = tf.Variable(
                    tf.random_uniform(
                        [self.vocab_size, self.embedding_size],
                        -1.0,
                        1.0,
                        dtype=tf.float32),
                    name='embedding')
            word_vectors = tf.nn.embedding_lookup(embedding, self.x_input)

            # Shape: [None, comment_size, embedding_size, 1]
            word_vectors_with_channel = tf.expand_dims(word_vectors, -1)

            pools = []
            for i, filter_size in enumerate(self.filter_sizes):
                with tf.name_scope('conv-{}'.format(filter_size)):
                    # Parameters
                    filter_shape = [filter_size, self.embedding_size, 1, self.num_filters]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name='W')
                    b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name='b')

                    # Convolution
                    conv = tf.nn.conv2d(
                            word_vectors_with_channel,
                            W,
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name='conv')

                    # Add nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

                    # Max-pool
                    pool = tf.nn.max_pool(
                            h,
                            ksize=[1, self.comment_size - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name='pool')
                    print(pool.shape)
                    pools.append(pool)

            # Combine pools
            num_filters_total = self.num_filters * len(self.filter_sizes)
            h_pool = tf.concat(pools, axis=3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout)

            # Add output
            with tf.name_scope('output'):
                num_classes = 2
                W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name='W')
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')
                self.predictor = tf.nn.xw_plus_b(h_drop, W, b, name='predictor')

            # Add loss
            with tf.name_scope('loss'):
                self.loss = tf.reduce_mean(
                        tf.nn.softmax_cross_entropy_with_logits(
                            logits=self.predictor,
                            labels=self.y_hot),
                        name='loss')
                tf.summary.scalar('loss', self.loss)

            # Optimize
            self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

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


