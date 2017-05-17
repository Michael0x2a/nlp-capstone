from typing import List, Dict, Tuple, Union, Any
import random
import time

import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore

from models.bag_of_words import BagOfWordsClassifier
from models.rnn_classifier import to_words, truncate_and_pad, make_vocab_mapping, vectorize_paragraph, WordId
import utils.file_manip as fmanip
from data_extraction.wikipedia import *


Params = Dict[str, Union[float, int]]
Nodes = Dict[str, Any]

def same_shape(item: Any, dims: Tuple[Any, ...]) -> bool:
    strs = ['?' if dim is None else str(dim) for dim in dims]
    actual = str(item.shape).replace(',)', ')')
    expected = '(' + ', '.join(strs) + ')'
    if actual != expected:
        raise AssertionError("Expected shape {}; got {}".format(expected, actual))
    else:
        return True


def main() -> None:
    # Note -- for now, I'm implementing this very simply as possible,
    # for the sake of testing.
    # Hyperparameters
    params = {
            #'vocab_size': 100000,
            'vocab_size': 30000,
            'embedding_size': 64,
            'n_hidden_layers': 512,
            'comment_size': 100,
            'batch_size': 50,
            'epoch_size': 20,
            'input_keep': 1.0,
            'output_keep': 0.7,
            'use_small': True,
    }  # type: Params
    log_dir = 'logs/filter/runx'

    # Train aggression classifier (naive, for now)
    #print("Loading aggression classifier")

    # Make autoencoder model

    fmanip.delete_folder(log_dir)

    # Train 
    (x_train, y_train), (x_test, y_test) = get_wikipedia_data(
            'attack', 
            'attack',
            use_small=params['use_small'])
    y_train_hard = [int(foo > 0.5) for foo in y_train]
    attack_clf = BagOfWordsClassifier()
    attack_clf.train(x_train, y_train_hard)
    print("Done training bag of words")

    x_final, x_lengths_orig, vocab_map, inv_vocab_map = prep_dataset(x_train, params)
    with tf.name_scope('rnn-filter-aggression'):
        x_input, x_lengths, input_keep, output_keep, is_train = build_model(params)
        sequence_loss, outputs, word_ids = build_seq2seq(input_keep, output_keep, x_input, x_lengths, is_train, params, attack_clf, inv_vocab_map)
        optimizer = build_optimizer(sequence_loss, params)

        summary = tf.summary.merge_all()
        logger = tf.summary.FileWriter(
                log_dir,
                graph=tf.get_default_graph())
        session = tf.Session(graph = tf.get_default_graph())

        nodes = {
                'x_input': x_input,
                'x_lengths': x_lengths,
                'input_keep': input_keep,
                'output_keep': output_keep,
                'is_train': is_train,
                'sequence_loss': sequence_loss,
                'outputs': outputs,
                'word_ids': word_ids,
                'optimizer': optimizer,
                'summary': summary,
                'logger': logger,
                'session': session,
        }


    print("Training")
    params['train'] = True
    train(x_final, x_lengths_orig, nodes, params)

    print("Predicting")
    # Warning: input size must be 100 for now (batch size is fixed; haven't fixed yet)
    # TODO: fix batch size thing
    
    interesting = []
    for i, sentence in enumerate(x_train):
        if y_train_hard[i]:
            interesting.append(sentence)
        if len(interesting) == params['batch_size']:
            break


    params['train'] = False
    out = predict(x_train[:params['batch_size']], vocab_map, nodes, params)
    for x, y in zip(x_train[:10], out[:10]):
        print(x)
        print()
        print('~~~')
        print()
        print(y)
        print()
        print('-------')
        print()
    print('INTERESTING:\n')

    out = predict(interesting, vocab_map, nodes, params)
    for x, y in zip(interesting[:10], out[:10]):
        print(x)
        print()
        print('~~~')
        print()
        print(y)
        print()
        print('-------')
        print()


def train(x_final: List[List[int]], x_lengths: List[int], nodes: Nodes, params: Params) -> None:
    nodes['session'].run(tf.initialize_all_variables())
    n_batches = int(len(x_final) // params['batch_size'])

    for i in range(int(params['epoch_size'])):
        indices = list(range(len(x_final)))
        random.shuffle(indices)
        x_lengths_new = [x_lengths[i] for i in indices]
        x_final_new = [x_final[i] for i in indices]

        train_epoch(i, n_batches, x_lengths, x_final, nodes, params)

def train_epoch(iteration: int,
                n_batches: int,
                x_lengths: List[int],
                xs: List[List[int]], 
                nodes: Nodes,
                params: Params) -> None:
    start = time.time()

    # Train on dataset
    batch_losses = []
    for batch_num in range(n_batches):
        start_idx = int(batch_num * params['batch_size'])
        end_idx = int((batch_num + 1) * params['batch_size'])

        x_batch = xs[start_idx: end_idx]
        x_len_batch = x_lengths[start_idx: end_idx]

        batch_data = {
                nodes['x_input']: x_batch, 
                nodes['x_lengths']: x_len_batch,
                nodes['input_keep']: params['input_keep'],
                nodes['output_keep']: params['output_keep'],
                nodes['is_train']: True,
        }

        s, l, _ = nodes['session'].run(
                [nodes['summary'], nodes['sequence_loss'], nodes['optimizer']], 
                feed_dict=batch_data)
        batch_losses.append(l)
    nodes['logger'].add_summary(s, iteration)
    delta = time.time() - start 
    print("Iteration {}, average loss = {:.6f} ({} batches), time elapsed = {:.3f}".format(
        iteration, sum(batch_losses) / len(batch_losses), n_batches, delta))

def predict(xs: List[str], vocab_map: Dict[str, int], nodes: Nodes, params: Params) -> Any:
    x_data_raw = [truncate_and_pad(to_words(x), int(params['comment_size'])) for x in xs]
    x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
    x_final = [vectorize_paragraph(vocab_map, para) for para in x_data_raw]
    batch_data = {
            nodes['x_input']: x_final,
            nodes['x_lengths']: x_lengths,
            nodes['input_keep']: 1.0,
            nodes['output_keep']: 1.0,
            nodes['is_train']: False,
    }
    ids = nodes['session'].run(nodes['word_ids'], feed_dict=batch_data)
    inv_vocab_map = {y: x for (x, y) in vocab_map.items()}
    out = []
    for index, line in enumerate(ids):
        out.append(' '.join([inv_vocab_map[i] for i in line][:x_lengths[index]]))
    return out



def build_model(params: Params) -> Tuple[Any, ...]:
    with tf.name_scope('inputs'):
        x_input = tf.placeholder(
                tf.int32, 
                shape=(params['batch_size'], params['comment_size']),
                name='x_input')
        x_lengths = tf.placeholder(
                tf.int32,
                shape=(params['batch_size'],),
                name='x_lengths')
        input_keep = tf.placeholder(
                tf.float32,
                shape=tuple(),
                name='input_keep')
        output_keep = tf.placeholder(
                tf.float32,
                shape=tuple(),
                name='output_keep')
        is_train = tf.placeholder(
                tf.bool,
                shape=tuple(),
                name='is_train')
    return (x_input, x_lengths, input_keep, output_keep, is_train)


def build_seq2seq(input_keep: Any, output_keep: Any, x_input: Any, x_lengths: Any, is_train: Any, params: Params, clf: Any, inv_vocab_map) -> Tuple[Any, ...]:
    assert input_keep is not None
    assert output_keep is not None
    assert x_input is not None
    with tf.name_scope('prediction'):
        assert same_shape(x_input, (params['batch_size'], params['comment_size']))

        # Shape is [vocab_size, embedding_size]
        '''embedding = tf.Variable(
                tf.random_uniform([params['vocab_size'], params['embedding_size']], -1.0, 1.0, dtype=tf.float32),
                dtype=tf.float32,
                name="embedding")'''

        # Make encoder/decoder inputs
        x_unstacked = tf.unstack(
                x_input, 
                axis=1)
        assert isinstance(x_unstacked, list) and same_shape(x_unstacked[0], (params['batch_size'],))

        encoder_inputs = x_unstacked
        decoder_inputs = [tf.zeros_like(encoder_inputs[0], dtype=tf.int32, name="GO")] + encoder_inputs[:-1]

        # Make RNN

        cell = tf.contrib.rnn.BasicLSTMCell(params['n_hidden_layers'])
        '''cell = tf.contrib.rnn.MultiRNNCell([
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(params['n_hidden_layers'])),
            tf.contrib.rnn.DropoutWrapper(
                tf.contrib.rnn.BasicLSTMCell(params['n_hidden_layers'])),
        ])'''

        out_weight = tf.Variable(
                tf.random_uniform([params['n_hidden_layers'], params['vocab_size']], -1.0, 1.0, dtype=tf.float32),
                name='out_weight')
        out_bias = tf.Variable(
                tf.random_uniform([params['vocab_size']], -1.0, 1.0, dtype=tf.float32),
                name='out_bias')

        outputs_proj, states = tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
                encoder_inputs=encoder_inputs,
                decoder_inputs=decoder_inputs,
                cell=cell,
                num_encoder_symbols=params['vocab_size'],
                num_decoder_symbols=params['vocab_size'],
                embedding_size=params['embedding_size'],
                output_projection=(out_weight, out_bias),
                feed_previous=tf.logical_not(is_train),
                dtype=tf.float32)

        outputs = [tf.matmul(out, out_weight) + out_bias for out in outputs_proj]
        outputs_stacked = tf.stack(outputs, axis=1)
        word_ids = tf.argmax(outputs_stacked, axis=2)

        # Make loss function. Note: we're training an autoencoder, so
        # train and hit the original sentence
        def test(word_ids):
            out = []
            for index, line in enumerate(word_ids):
                toks = [inv_vocab_map.get(i, '$UNK') for i in line]

                new_toks = []
                for t in toks:
                    new_toks.append(t)
                    if t == '$END':
                        break
                out.append(' '.join(new_toks))
            return np.asarray([1 - a for a in clf.predict(out)], dtype=np.float32)

        seq_loss = softmax_xent_loss_sequence(
                logits=outputs_stacked,
                targets=x_input,
                seq_len=x_lengths,
                max_seq_len=params['comment_size'])
        agg_loss = tf.reduce_mean(tf.py_func(test, [word_ids], [tf.float32]))
        print(agg_loss.shape)

        #seqW = tf.Variable(
        #        tf.random_uniform([], -1.0, 1.0, dtype=tf.float32),
        #        name='seqW')
        seqW = 0.6
        loss = tf.add(tf.multiply(seq_loss, seqW), 
               tf.multiply(agg_loss, tf.subtract(1.0, seqW)))

        tf.summary.scalar("sequence_loss", loss)

        return loss, outputs_stacked, word_ids


def softmax_xent_loss_sequence(logits, targets, seq_len, max_seq_len, reduce_mean=True):
    ones = tf.ones_like(seq_len)
    ones_float = tf.ones_like(seq_len, dtype=tf.float32)
    zeros = ones_float * 0
    weight_list = [
            tf.where(
                tf.less_equal(
                    ones * i, seq_len - 1),
                ones_float, zeros)
            for i in range(max_seq_len)]
    weights = tf.transpose(tf.stack(weight_list))

    xent = tf.contrib.seq2seq.sequence_loss(
            logits, targets, weights, average_across_batch=reduce_mean)

    return xent

def build_optimizer(sequence_loss: Any, params: Params) -> Any:
    # TODO: Incorporate other model
    with tf.name_scope('evaluation'):
        optimizer = tf.train.AdamOptimizer().minimize(sequence_loss)
        return optimizer



# TODO: Refactor instead of copying and pasting
def prep_dataset(xs: List[str], params: Params) -> Tuple[List[List[WordId]], List[int], Dict[str, WordId], Dict[WordId, str]]: 
    x_data_raw = [truncate_and_pad(to_words(x), int(params['comment_size'])) for x in xs]
    x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
    vocab_map = make_vocab_mapping(x_data_raw, int(params['vocab_size']))
    inv_vocab_map = {value: key for (key, value) in vocab_map.items()}
    x_final = [vectorize_paragraph(vocab_map, para) for para in x_data_raw]
    return x_final, x_lengths, vocab_map, inv_vocab_map

Primitive = Union[int, float, str, bool]
Data = Tuple[List[str], List[float]]

# TODO: Refactor, instead of copying from cmd.py
def get_wikipedia_data(category: str = None,
                       attribute: str = None,
                       use_dev: bool = True,
                       use_small: bool = False) -> Tuple[Data, Data]:
    funcs = {
            'attack': load_attack_data,
            'toxicity': load_toxicity_data,
            'aggression': load_aggression_data
    }

    assert category is not None and category in funcs
    if attribute is None:
        attribute = category
    assert attribute is not None

    def extract_data(comments: AttackData) -> Data:
        x_values = []
        y_values = []
        for comment in comments:
            x_values.append(comment.comment)
            cls = getattr(comment.average, attribute)  # type: ignore
            y_values.append(cls)
        return x_values, y_values

    train_data, dev_data, test_data = funcs[category](small=use_small)  # type: ignore

    train = extract_data(train_data)
    if use_dev:
        test = extract_data(dev_data)
    else:
        test = extract_data(test_data)
        
    return train, test

if __name__ == '__main__':
    main()

