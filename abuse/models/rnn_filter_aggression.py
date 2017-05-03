from typing import List, Dict, Tuple, Union, Any

import tensorflow as tf  # type: ignore
import numpy as np  # type: ignore

from models.bag_of_words import BagOfWordsClassifier
from models.rnn_classifier import to_words, truncate_and_pad, make_vocab_mapping, vectorize_paragraph, WordId
import utils.file_manip as fmanip


Params = Dict[str, Union[float, int]]

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
            'vocab_size': 140000,
            'embedding_size': 32,
            'n_hidden_layers': 64,
            'comment_size': 100,
            'batch_size': 200,
    }  # type: Params

    # Train aggression classifier (naive, for now)
    #print("Loading aggression classifier")
    #aggression_clf = BagOfWordsClassifier.restore_from_saved(
    #        fmanip.join("core_models", "aggression_bag_of_words"))

    # Make autoencoder model
    with tf.name_scope('rnn-filter-aggression'):
        x_input, x_lengths, input_keep, output_keep = build_model(params)
        build_seq2seq(input_keep, output_keep, x_input, params)


    # Train 
    x_final, vocab_map, inv_vocab_map = prep_dataset(params)

    # Test


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
    return (x_input, x_lengths, input_keep, output_keep)

def build_seq2seq(input_keep: Any, output_keep: Any, x_input: Any, params: Params) -> None:
    assert input_keep is not None
    assert output_keep is not None
    assert x_input is not None
    with tf.name_scope('prediction'):
        assert same_shape(x_input, (params['batch_size'], params['comment_size']))

        # Make encoder/decoder inputs
        x_unstacked = tf.unstack(
                x_input, 
                axis=1)
        assert isinstance(x_unstacked, list) and same_shape(x_unstacked[0], (params['batch_size'],))

        # Make projection output
        out_weight = tf.Variable(
                tf.random_uniform([params['n_hidden_layers'], params['vocab_size']], -1.0, 1.0, dtype=tf.float32),
                name='out_weight')
        out_bias = tf.Variable(
                tf.random_uniform([params['vocab_size']], -1.0, 1.0, dtype=tf.float32),
                name='out_bias')


        # Make RNN
        cell = tf.contrib.rnn.BasicLSTMCell(params['n_hidden_layers'])
        outputs, states = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
                encoder_inputs=x_unstacked,
                decoder_inputs=x_unstacked,
                cell=cell,
                num_encoder_symbols=params['vocab_size'],
                num_decoder_symbols=params['vocab_size'],
                embedding_size=params['embedding_size'],
                #output_projection=(out_weight, out_bias),
                dtype=tf.float32)
        print(len(outputs))
        print(outputs[0].shape)  # [comment_size, n_hidden_layers]

        out_sentences = [tf.argmax(out, axis=1) for out in outputs]

        # Make loss function (for now, just based on sentence similarity)
        #labels = tf.reshape(labels, [-1, 1])
        loss = tf.nn.sampled_softmax_loss(
                weights=tf.transpose(out_weight),
                biases=out_bias,
                labels=
                
                w_t, b, inputs, labels, num_samples,
        #    self.target_vocab_size)
        #softmax_loss_function = sampled_loss


def prep_dataset(params: Params) -> Tuple[List[List[WordId]], Dict[str, WordId], Dict[WordId, str]]: 
    x_data_raw = [truncate_and_pad(to_words(x), params['comment_size']) for x in xs]
    x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
    vocab_map = make_vocab_mapping(x_data_raw, params['vocab_size'])
    inv_vocab_map = {value: key for (key, value) in vocab_map.items()}
    x_final = [vectorize_paragraph(vocab_map, para) for para in x_data_raw]
    return x_final, vocab_map, inv_vocab_map

if __name__ == '__main__':
    main()
