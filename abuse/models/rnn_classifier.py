from typing import List, Tuple
import os.path
import shutil

import numpy as np
import tensorflow as tf
import nltk
import sklearn.metrics

from parsing import load_raw_data
from custom_types import *

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

def extract_is_attack(comments: List[Comment],
                      max_length: int) -> Tuple[List[Paragraph], List[Label]]:
    x_values = []
    y_values = []
    for comment in comments:
        # TODO: Better-tokenize sentence
        x_values.append(truncate_and_pad(
                nltk.word_tokenize(comment.comment),
                max_length))

        # TODO: Train in more granularity? Besides IS_ATTACK and IS_OK,
        # add 'maybe' labels?
        # TODO: There's a lot of unnecessary casting between bools and ints here...
        # Figure out a way to rephrase w/o losing readability
        cls = IS_ATTACK if comment.average.attack > 0.5 else IS_OK
        y_values.append([1 if cls == IS_OK else 0, 1 if cls == IS_ATTACK else 0])
    return x_values, y_values


def make_vocab_mapping(x: List[Paragraph]) -> Dict[str, WordId]:
    out = {}
    count = 0
    for paragraph in x:
        for word in paragraph:
            if word in out:
                continue
            out[word] = count
            count += 1
    return out 

def vectorize_paragraph(vocab_map: Dict[str, WordId], para: Paragraph) -> ParagraphVec:
    return [vocab_map[word] for word in para]


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


def main() -> None:
    print("Start...")

    # Hyperparameters (?)
    paragraph_size = 100
    batch_size = 120  # Arbitrary choice
    data_src_path = 'data/wikipedia-detox-data-v6'
    training_iters = 10
    n_hidden_layers = 5
    embedding_size = 32

    # Constants
    log_dir = os.path.join("logs", "rnn_classifier")
    n_classes = 2  # For now

    # Delete log folder
    shutil.rmtree(log_dir, ignore_errors=True)

    # Load data
    print("Loading data...")
    train_data, dev_data, test_data = load_raw_data(data_src_path)

    # Take data, and split into x/y pairs, truncating input to
    # length 500 (including start and end tokens. Shorter sentences
    # are padded)
    x_train_raw, y_train = extract_is_attack(train_data, paragraph_size)
    x_dev_raw, y_dev = extract_is_attack(dev_data, paragraph_size)

    # Make vocab list
    # TODO: Refine to handle things like UNK; allow dev data to be
    # unknown beforehand
    print("Vectorizing and prepping...")
    vocab_map = make_vocab_mapping(x_train_raw + x_dev_raw)
    vocab_size = len(vocab_map)

    # Convert paragraphs into real_valued vector
    x_train = [vectorize_paragraph(vocab_map, para) for para in x_train_raw]
    x_dev = [vectorize_paragraph(vocab_map, para) for para in x_dev_raw]

    ### 

    # Begin defining tensorflow model

    # Placeholders for input
    print("Building model...")
    with tf.name_scope('inputs'):
        paragraphs_placeholder = tf.placeholder(
                tf.int32, 
                shape=(None, paragraph_size),
                name='x_input')
        labels_placeholder = tf.placeholder(
                tf.float32,
                shape=(None, n_classes),
                name='y_input')

    # Define predictor, loss, and optimizer
    with tf.name_scope('prediction'):
        # Make embedding vector for words
        # Shape is [?, paragraph_size, embedding_size]
        embedding = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="embedding")
        x_data = tf.nn.embedding_lookup(embedding, paragraphs_placeholder)

        prediction = make_bidirectional_rnn(
                x_data, paragraph_size, n_hidden_layers, n_classes)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=prediction, 
                labels=labels_placeholder),
                name='cost')
        optimizer = tf.train.AdamOptimizer().minimize(cost)

    # Evaluation
    with tf.name_scope('evaluation'):
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels_placeholder, 1))
        accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32),
                name='accuracy')

        y_pred = tf.argmax(prediction, 1)

    # Log data for tensorboard
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)
    summaries = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())

    # Add variable initializer op
    init = tf.global_variables_initializer()

    # Saver for training checkpoints: TODO
    #saver = tf.train.Saver()

    # Begin actually training!
    print("Training...")
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        n_batches = len(x_train) // batch_size

        session.run(init)
        for i in range(training_iters):
            # Train on dataset
            for batch_num in range(n_batches):
                start_idx = batch_num * batch_size
                end_idx = (batch_num + 1) * batch_size

                x_batch = x_train[start_idx: end_idx]
                y_batch = y_train[start_idx: end_idx]

                batch_data = {
                    paragraphs_placeholder: x_batch,
                    labels_placeholder: y_batch
                }

                session.run(optimizer, feed_dict=batch_data)

            # Report results, using last x_batch and y_batch
            summary_data, acc, loss = session.run([summaries, accuracy, cost], feed_dict=batch_data)

            print("Iteration {}, batch loss={:.6f}, batch accuracy={:.6f}".format(
                i, loss, acc))
            train_writer.add_summary(summary_data, i)

            # Evaluate on dev
            print("    Evaluating on dev...")
            batch_dev_data = {
                paragraphs_placeholder: x_dev,
                labels_placeholder: y_dev
            }
            acc, y_pred_val = session.run([accuracy, y_pred], feed_dict=batch_dev_data)
            y_true = np.argmax(np.asarray(y_dev), 1)

            print("    Accuracy:   {:.6f}".format(acc))
            print("    Accuracy 2: {:.6f}".format(sklearn.metrics.accuracy_score(y_true, y_pred_val)))
            print("    Precision:  {:.6f}".format(sklearn.metrics.precision_score(y_true, y_pred_val)))
            print("    Recall:     {:.6f}".format(sklearn.metrics.recall_score(y_true, y_pred_val)))
            print("    F1 score:   {:.6f}".format(sklearn.metrics.f1_score(y_true, y_pred_val)))
            print("    AUC score:  {}".format(sklearn.metrics.roc_auc_score(y_true, y_pred_val)))
            print("    Confusion matrix:")
            print(sklearn.metrics.confusion_matrix(y_true, y_pred_val))
            print()

        print("Done training!")



    #session = tf.Session()

    # SummaryWriter for outputting summaries and the graph
    #summary_writer = tf.summary.FileWriter(log_dir, session.graph)
    ####

if __name__ == '__main__':
    main()


