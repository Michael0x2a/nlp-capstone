from typing import List, Tuple
from parsing import import_raw_data
from custom_types import *

# Labels
IS_ATTACK = 1
IS_OK = 0

# Type aliases, for readability
Paragraph = List[str]
Label = int

def truncate_and_pad(paragraph: Paragraph, max_length: int) -> Paragraph:
    # Subtract 2 so we have space for the start and end tokens
    length = min(len(paragraph), max_length - 2)
    if length < max_length - 2:
        padding = max_length - 2 - length
    else:
        passing = 0

    out = ["$START"] + paragraph[:length] + ["$END"] + (["$PADDING"] * padding)
    return out

def extract_is_attack(comments: List[Comment],
                      max_length: int) -> Tuple[List[Paragraph], List[Label]]:
    x_values = []
    y_values = []
    for comment in comments:
        # TODO: Better-tokenize sentence
        x_values.append(truncate_and_pad(
                comment.comment.split(" "), 
                max_length))

        # TODO: Train in more granularity? Besides IS_ATTACK and IS_OK,
        # add 'maybe' labels?
        y_values.append(IS_ATTACK if comment.average.attack > 0.5 else IS_OK)
    return x_values, y_values


def main():
    # Hyperparameters (?)
    paragraph_size = 500
    batch_size = None  # (keep it flexible)
    log_dir = "logs"

    # Load data
    train_data, dev_data, test_data = import_raw_data()

    # Take data, and split into x/y pairs, truncating input to
    # length 500 (including start and end tokens. Shorter sentences
    # are padded)
    x_train_raw, y_train = extract_is_attack(train_data)

    # Convert paragraphs into real_valued vector
    # TODO

    # Begin defining tensorflow model
    with tf.Graph().as_default():
        # Placeholders for input
        paragraphs_placeholder = tf.placeholder(
                tf.float32, 
                shape=(batch_size, paragraph_size))
        labels_placeholder = tf.placeholder(
                tf.int32,
                shape=(batch_size,))

        # TODO: Define loss function
        # TODO: Add train code
        lstm_size = # ???
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        state = tf.zeros([batch_size, lstm.state_size])


        # TODO: Add evaluation code

        # Summary tensor
        summary = tf.summary.merge_all()

        # Add variable initializer op
        init = tf.global_variables_initializer()

        # Saver for training checkpoints
        saver = tf.train.Saver()

        # Session for running ops on a graph
        session = tf.Session()

        # SummaryWriter for outputting summaries and the graph
        summary_writer = tf.summary.FileWriter(log_dir, session.graph)

        ####

        # Initialize variables
        sess.run(init)

        # Training loop
        for step in xrange(1000):
            start_time = time.time()

            # Feed in data
            feed_dict = images_placeholder.






