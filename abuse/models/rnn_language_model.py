from typing import List, Tuple, Optional, Dict, cast, TypeVar, Generator
import os.path
import shutil
import json
import time  # type: ignore
import random
from collections import Counter
import re
from math import ceil

import numpy as np  # type: ignore
import tensorflow as tf  # type: ignore
import nltk  # type: ignore
import nltk.corpus  # type: ignore
import sklearn.metrics  # type: ignore

import utils.file_manip as fmanip
from data_extraction.wikipedia import * 
from custom_types import *

from models.model import Model
from utils.unks import truncate_and_pad, to_words, vectorize_paragraph, Paragraph, WordId, ParagraphVec, Label, VocabMap

#b

print("Done loading imports")

# Labels
IS_ATTACK = 1
IS_OK = 0

# def to_word_array(xs: List[str]) -> List[List[str]]:
#     words = [to_words(x) for x in xs]
#     max_len = max(len(x) for x in words)
#     print("max length:", max_len)
#     return [truncate_and_pad(x, max_len) for x in words]

def add_markers(xs: List[Paragraph]):
    return [["$START"] + x + ["$END"] for x in xs]

def pad(xs: List[List[int]], lengths: List[int]) -> Tuple[List[Paragraph], List[List[bool]]]:
    max_len = max(lengths)
    # padding has id 1
    # padded = [x + [1] * (max_len - len(x)) for x in xs]

    padded = np.ones((len(xs), max_len), dtype=np.int32)
    for i, (x, length) in enumerate(zip(xs, lengths)):
        padded[i,:length] = x

    non_padding = np.arange(max_len) < lengths.reshape(-1, 1)
    return padded, non_padding

def make_vocab_mapping(x: List[Paragraph],
                       max_vocab_size: Optional[int] = None) -> Dict[str, WordId]:
    freqs = Counter()  # type: Counter[str]
    words_set = set()
    for paragraph in x:
        for word in paragraph:
            freqs[word] = freqs.get(word, 0) + 1
            words_set.add(word)
    word_to_id = {'$UNK': 0, '$PADDING':1}
    id_to_word = ['$UNK', '$PADDING']
    id = 2
    if max_vocab_size is None:
        max_vocab_size = len(freqs)
    print("Actual vocab size: {}".format(len(freqs)))
    for key, num in freqs.most_common(max_vocab_size - 2):
        word_to_id[key] = id
        id_to_word.append(key)
        id += 1
    with open("unks.txt", "w") as stream:
        for word in words_set:
            if word not in word_to_id:
                stream.write(word)
                stream.write("\n")
    return word_to_id, np.array(id_to_word)

T = TypeVar("T")
def chunks(l: List[T], n: int) -> Generator[List[T], None, None]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class RnnLanguageModel(Model[str]):
    base_log_dir = "runs/rnn_lm/run{}"

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
                       epsilon: float = 1e-08,
                       k_top_words: int = 50) -> None:

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
        self.k_top_words = k_top_words

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
        self.cls_loss = None    # type: Any
        self.lm_loss = None     # type: Any
        self.optimizer = None   # type: Any
        self.summary = None     # type: Any
        self.output = None      # type: Any
        self.output_prob = None # type: Any
        self.words = None       # type: Any
        self.top_probs = None   # type: Any
        self.top_words = None   # type: Any
        self.words_probs = None # type: Any
        self.cell_outputs = None# type: Any
        self.word_logits = None # type: Any
        self.init = None        # type: Any
        self.logger = None      # type: Any
        self.session = None     # type: Any

        self.vocab_map = None   # type: Optional[Dict[str, WordId]]
        self.vocab_map_backwards = None # type: Optional[Dict[WordId, str]]

        self.run_num = self._get_next_run_num()
        self._build_model()
        super().__init__(restore_from, run_num)

    def _get_parameters(self) -> Dict[str, Any]:
        return {
                'comment_size': self.comment_size,
                'batch_size': self.batch_size,
                'epoch_size': self.epoch_size,
                'n_hidden_layers': self.n_hidden_layers,
                'vocab_size': self.vocab_size,
                'embedding_size': self.embedding_size,
                'n_classes': self.n_classes,
                'input_keep_prob': self.input_keep_prob,
                'output_keep_prob': self.output_keep_prob,
                'learning_rate': self.learning_rate,
                'beta1': self.beta1,
                'beta2': self.beta2,
                'epsilon': self.epsilon,
                'k_top_words': self.k_top_words,
        }

    def _save_model(self, path: str) -> None:
        with open(fmanip.join(path, 'vocab_map.json'), 'w') as f:
            json.dump(self.vocab_map, f)
        # np.save(fmanip.join(path, 'vocab_map_backwards.npy'), self.vocab_map_backwards)
        with open(fmanip.join(path, 'vocab_map_backwards.json'), 'w') as f:
            json.dump(self.vocab_map_backwards.tolist(), f)

        saver = tf.train.Saver()

        saver.save(self.session, fmanip.join(path, 'model'))
        # tf.train.export_meta_graph(filename=fmanip.join(path, 'tensorflow_graph.meta'))

    def _restore_model(self, path: str) -> None:
        with open(fmanip.join(path, 'vocab_map.json'), 'r') as f:
            self.vocab_map = json.load(f)
        try:
            with open(fmanip.join(path, 'vocab_map_backwards.json'), 'r') as f:
                self.vocab_map_backwards = np.array(json.load(f))
            # np.save(fmanip.join(path, 'vocab_map_backwards.npy'), self.vocab_map_backwards)
        except IOError:
            self.vocab_map_backwards = np.load(fmanip.join(path, 'vocab_map_backwards.npy'))
            with open(fmanip.join(path, 'vocab_map_backwards.json'), 'w') as f:
                json.dump(self.vocab_map_backwards.tolist(), f)

        self.session = tf.Session(graph = tf.get_default_graph())
        # saver = tf.train.import_meta_graph(fmanip.join(path, 'tensorflow_graph.meta'))
        saver = tf.train.Saver()
        saver.restore(self.session, fmanip.join(path, 'model'))

        # self._assert_all_setup()

    def _build_model(self) -> None:
        '''Builds the model, using the currently set params.'''
        with tf.name_scope('rnn-classifier'):
            self._build_input()
            self._build_predictor()
            self._build_evaluator()

            print('output_shape', self.output.shape)

            self.summary = tf.summary.merge_all()
            self.init = tf.global_variables_initializer()

        #self.session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        self.session = tf.Session(graph = tf.get_default_graph())

    def _build_input(self) -> None:
        with tf.name_scope('inputs'):
            self.x_input = tf.placeholder(
                    tf.int32, 
                    shape=(None, None), # self.comment_size
                    name='x_input')
            self.x_mask = tf.placeholder(
                    tf.bool,
                    shape=(None, None),
                    name='x_mask')
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
            x_shape = tf.shape(self.x_input)
            self.cls_loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.predictor, 
                        labels=self.y_hot,),
                    name='classifier_loss')

            self.log_perplexity = tf.contrib.seq2seq.sequence_loss(
                    logits=self.word_logits,
                    targets=self.x_input[:,1:-1],
                    weights=tf.cast(self.x_mask[:,2:], tf.float32), # drop first 2 to account for cutting out start/end
                    name="language_model_loss")
            self.loss = self.cls_loss + self.log_perplexity / 4 # hardcode weighting for now

            self.optimizer = tf.train.AdamOptimizer(
                    learning_rate=self.learning_rate,
                    beta1=self.beta1,
                    beta2=self.beta2,
                    epsilon=self.epsilon).minimize(self.loss)

            tf.summary.scalar('classifier_loss', self.cls_loss)
            tf.summary.scalar('language_model_loss', self.log_perplexity)
            tf.summary.scalar('loss', self.loss)

    def _build_evaluator(self) -> None:
        with tf.name_scope('evaluation'):
            correct_prediction = tf.equal(
                    tf.argmax(self.predictor, 1),
                    tf.argmax(self.y_hot, 1))
            accuracy = tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32),
                    name='accuracy')
            tf.summary.scalar('classifier_accuracy', accuracy)

            self.output = tf.argmax(self.predictor, 1, name='output')
            self.output_prob = tf.nn.softmax(self.predictor, name='output_prob')

            # lm accuracy not useful
            # correct_words = tf.equal(
            #         tf.cast(tf.argmax(self.word_logits, axis=2), tf.int32),
            #         self.x_input[:,1:-1])
            # lm_accuracy = tf.reduce_mean(
            #         tf.cast(correct_words, tf.float32),
            #         name='words_accuracy')
            # tf.summary.scalar('language_model_accuracy', lm_accuracy)

            self.words = tf.argmax(self.word_logits, axis=2, name='words')
            self.word_log_probs = tf.nn.log_softmax(self.word_logits, name='words_log_probs')
            self.words_probs = tf.exp(self.word_log_probs, name='words_probs')

            self.top_log_probs, self.top_words =\
                    tf.nn.top_k(self.word_log_probs, self.k_top_words, name='top_words')
            self.top_probs = tf.exp(self.top_log_probs, name='top_probs')


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
                    tf.zeros([self.n_classes], dtype=tf.float32),
                    dtype=tf.float32,
                    name='output_bias')
            lm_weight = tf.Variable(
                    tf.random_normal([self.n_hidden_layers * 2, self.vocab_size], dtype=tf.float32),
                    name='language_model_weight')
            lm_bias = tf.Variable(
                    tf.zeros([self.vocab_size], dtype=tf.float32),
                    name='language_model_bias')


            # Defining the bidirectional rnn
            # layer = x_unstacked
            for i in range(1):
                with tf.name_scope('layer_{}'.format(i)):
                    with tf.variable_scope('forwards'):
                        forwards_cell = tf.contrib.rnn.DropoutWrapper(
                                tf.contrib.rnn.BasicLSTMCell(self.n_hidden_layers),
                                input_keep_prob=self.input_keep,
                                output_keep_prob=self.output_keep)
                    with tf.variable_scope('backwards'):
                        backwards_cell = tf.contrib.rnn.DropoutWrapper(
                                tf.contrib.rnn.BasicLSTMCell(self.n_hidden_layers),
                                input_keep_prob=self.input_keep,
                                output_keep_prob=self.output_keep)
                    #forwards_cell = tf.contrib.rnn.GRUCell(self.n_hidden_layers)
                    #backwards_cell = tf.contrib.rnn.GRUCell(self.n_hidden_layers)

                    # forwards_cells = [tf.contrib.rnn.DropoutWrapper(
                    #         tf.contrib.rnn.BasicLSTMCell(self.n_hidden_layers),
                    #         input_keep_prob=self.input_keep,
                    #         output_keep_prob=self.output_keep) for i in range(2)]
                    # backwards_cells = [tf.contrib.rnn.DropoutWrapper(
                    #         tf.contrib.rnn.BasicLSTMCell(self.n_hidden_layers),
                    #         input_keep_prob=self.input_keep,
                    #         output_keep_prob=self.output_keep) for i in range(2)]
                    
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
            # foo = tf.concat([fw_state.h,bw_state.h], axis=1)
            foo = tf.layers.batch_normalization(tf.concat([fw_state.h,bw_state.h], axis=1))
            # (batch_size, 2*hidden_size)
            prediction = tf.matmul(foo, output_weight) + output_bias

            # concat states from either side of words
            self.cell_outputs = tf.concat([fw_outputs[:,:-2], bw_outputs[:,2:]], axis=2)
            # tf.matmul doesn't work with different ranks. yay.
            # [batch_size, max_time-2, output_size*2] . [output_size*2, vocab_size]
            #      -> [batch_size, max_time-2, vocab_size]
            matmul = tf.einsum('ijk,kn->ijn', self.cell_outputs, lm_weight)
            self.word_logits = matmul + lm_bias
            return prediction

    def train(self, xs: List[str], ys: List[int], **params: Any) -> None:
        '''Trains the model. The expectation is that this method is called
        exactly once.'''
        if len(params) != 0:
            raise Exception("RNN does not take in any extra params to train")


        self.logger = tf.summary.FileWriter(self._get_log_dir(), graph=tf.get_default_graph())

        ys = np.array(ys)
        bin_edges = np.concatenate([
                np.arange(0,200,20),
                np.arange(200,1000,100),
                np.arange(1000,1500,500),
                np.arange(1500,3000,500)])  # (max is 10000)
        words = add_markers(to_words(x) for x in xs)
        lengths = np.fromiter((len(x) for x in words), dtype=np.int, count=len(xs))
        self.vocab_map, self.vocab_map_backwards = make_vocab_mapping(words, self.vocab_size)
        ids = np.array([vectorize_paragraph(self.vocab_map, x) for x in words])
        # (can't create object array from iterator, apparently)

        placements = np.digitize(lengths, bin_edges) - 1
        bins = [np.where(placements==i)[0] for i in range(len(bin_edges))]

        # x_data_raw = to_word_array(xs)
        # x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
        # self.vocab_map = make_vocab_mapping(x_data_raw, self.vocab_size)
        # x_final = [vectorize_paragraph(self.vocab_map, para) for para in x_data_raw]

        # n_batches = len(x_final) // self.batch_size

        # self._assert_all_setup()

        self.session.run(self.init)
        for i in range(self.epoch_size):
            # indices = list(range(len(x_final)))
            # random.shuffle(indices)
            # x_lengths_new = [x_lengths[i] for i in indices]
            # x_final_new = [x_final[i] for i in indices]
            # ys_new = [ys[i] for i in indices]

            self.train_epoch(i, bins, self.batch_size, lengths, ids, ys)

    def train_epoch(self, iteration: int,
                          bins: List[List[int]],
                          batch_size: int,
                          lengths: List[int],
                          xs: List[List[int]],
                          ys: List[int]) -> None:
        start = time.time()
        batch_count = sum(ceil(len(bin) / batch_size) for bin in bins)
        losses = 0.0
        i = 0
        random.shuffle(bins)

        # Train on dataset
        for bin in bins:
            random.shuffle(bin)
            for batch in chunks(bin, batch_size):
                if i % 1 == 0:
                    print("Training batch", i, "size", len(batch), "len", len(xs[batch[0]]), "         ", end="\r")
                
                xs_batch, xs_mask = pad(xs[batch], lengths[batch])
                batch_data = {
                        self.x_lengths: lengths[batch],
                        self.x_input: xs_batch,
                        self.x_mask: xs_mask,
                        self.y_input: ys[batch],
                        self.input_keep: self.input_keep_prob,
                        self.output_keep: self.output_keep_prob,
                }

                summary_data, batch_loss, _ = self.session.run(
                        [self.summary, self.loss, self.optimizer], 
                        feed_dict=batch_data)
                losses += batch_loss
                self.logger.add_summary(summary_data, iteration*batch_count + i)
                i += 1

        # Report results, using last x_batch and y_batch
        delta = time.time() - start 
        print("Iteration {}, avg batch loss = {:.6f}, num batches = {}, time elapsed = {:.3f}".format(
            iteration,
            losses / i,
            i,
            delta))

    def predict(self, xs: List[str]) -> List[List[float]]:
        # assert self.vocab_map is not None
        # xs = np.array(xs)
        out = np.empty([len(xs), self.n_classes], dtype=np.float32)
        return self._run(self.output_prob, xs, out)
        # bin_edges = np.concatenate([
        #         np.arange(0,200,20),
        #         np.arange(200,1000,100),
        #         np.arange(1000,1500,500),
        #         np.arange(1500,3000,500)])
        # words = add_markers(to_words(x) for x in xs)
        # lengths = np.fromiter((len(x) for x in words), dtype=np.int, count=len(xs))
        # ids = np.array([vectorize_paragraph(self.vocab_map, x) for x in words])
        # # (can't create object array from iterator, apparently)

        # placements = np.digitize(lengths, bin_edges) - 1
        # bins = [np.where(placements==i)[0] for i in range(len(bin_edges))]

        # for batch in bins:
        #     print("Testing batch", i, "size", len(batch), "len", len(xs[batch[0]]), end="\r")
            
        #     batch_data = {
        #             self.x_lengths: lengths[batch],
        #             self.x_input: pad(xs[batch], lengths[batch]),
        #             self.input_keep: 1.0,
        #             self.output_keep: 1.0,
        #     }

        #     out[batch] = self.session.run([self.output_prob], feed_dict=batch_data)
        # return out

    def _run(self, tensor: any, xs: List[str], out: List[any]) -> List[any]:
        assert self.vocab_map is not None
        bin_edges = np.concatenate([
                np.arange(0,200,20),
                np.arange(200,1000,100),
                np.arange(1000,1500,500),
                np.arange(1500,3000,500)])
        words = add_markers(to_words(x) for x in xs)
        lengths = np.fromiter((len(x) for x in words), dtype=np.int, count=len(xs))
        ids = np.array([vectorize_paragraph(self.vocab_map, x) for x in words])
        # (can't create object array from iterator, apparently)

        placements = np.digitize(lengths, bin_edges) - 1
        bins = [np.where(placements==i)[0] for i in range(len(bin_edges))]

        i = 0
        for batch in bins:
            if not batch.size:
                continue
            print("Testing batch", i, "size", len(batch), "len", len(xs[batch[0]]), "         ", end="\r")
            i += 1

            xs_batch, xs_mask = pad(ids[batch], lengths[batch])
            batch_data = {
                    self.x_lengths: lengths[batch],
                    self.x_input: xs_batch,
                    self.x_mask: xs_mask,
                    self.input_keep: 1.0,
                    self.output_keep: 1.0,
            }

            result = self.session.run(tensor, feed_dict=batch_data)
            if isinstance(tensor, list):
                for t, r in zip(out, result):
                    t[batch] = r
            else:
                out[batch] = result
        print("                                                     ")
        return out

    def _run_batch(self, tensor: any, xs: List[str]) -> any:
        words = add_markers(to_words(x) for x in xs)
        lengths = np.fromiter((len(x) for x in words), dtype=np.int, count=len(xs))
        ids = [vectorize_paragraph(self.vocab_map, x) for x in words]
        # (can't create object array from iterator, apparently)

        xs_batch, xs_mask = pad(ids, lengths)
        batch_data = {
                self.x_lengths: lengths,
                self.x_input: xs_batch,
                self.x_mask: xs_mask,
                self.input_keep: 1.0,
                self.output_keep: 1.0,
        }
        return self.session.run(tensor, feed_dict=batch_data)

    def top1(self, xs: List[str]) -> List[List[str]]:
        words = self._run_batch(self.words, xs)
        return self.vocab_map_backwards[words]

    def topk(self, xs: List[str]) -> Tuple[List[List[List[float]]], List[List[List[str]]]]:
        probs, words = self._run_batch([self.top_probs, self.top_words], xs)
        return probs, self.vocab_map_backwards[words]

    def perplexity(self, xs: List[str]) -> float:
        return np.exp(self._run_batch(self.log_perplexity, xs))

    def _run_batch_tokenized(self, tensor: any, xs: List[List[str]]) -> any:
        words = add_markers(x for x in xs)
        lengths = np.fromiter((len(x) for x in words), dtype=np.int, count=len(xs))
        ids = [vectorize_paragraph(self.vocab_map, x) for x in words]
        # (can't create object array from iterator, apparently)

        xs_batch, xs_mask = pad(ids, lengths)
        batch_data = {
                self.x_lengths: lengths,
                self.x_input: xs_batch,
                self.x_mask: xs_mask,
                self.input_keep: 1.0,
                self.output_keep: 1.0,
        }
        return self.session.run(tensor, feed_dict=batch_data)

    def perplexity_tokenized(self, xs: List[str]) -> float:
        return np.exp(self._run_batch_tokenized(self.log_perplexity, xs))
