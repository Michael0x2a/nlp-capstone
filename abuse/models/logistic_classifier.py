import os
import glob
import random
import numpy as np
from scipy.sparse import csr_matrix
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score

# default path to save things in
SUMMARY_PATH = "runs/lr{0}"

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class LogisticClassifier(object):
    def __init__(self, num_grams, base_summary_path=SUMMARY_PATH):
        self.num_grams = num_grams
        self.base_summary_path = base_summary_path

        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.graph = tf.Graph()
        self.reset()

        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.counts = tf.sparse_placeholder(tf.float32, [None, self.num_grams], "counts")
                self.target_labels = tf.placeholder(tf.bool, [None], "target_labels")

                self.pred_labels, self.logits, self.regularizer = self.build_model(self.counts)

                # I'm pretty sure this is necessary to add things to the summary
                self.loss = self.calculate_loss(self.logits, self.target_labels, self.regularizer)

                self.train_op = None

                self.summary = tf.summary.merge_all()

                self.saver = tf.train.Saver()


    def reset(self):
        self.session = tf.Session(graph=self.graph, config=self.config)
        i = 0
        while True:
            i += 1
            summary_path = self.base_summary_path.format(i)
            if not os.path.exists(summary_path) and not glob.glob(summary_path + "-*"):
                break
        self.summary_path = summary_path
        self.run_num = i
        self.it = 0
        self.epoch = 0

    def build_model(self, counts):
        with tf.name_scope("model"):
            W = tf.Variable(tf.random_normal([self.num_grams, 1]), name="weight")
            b = tf.Variable(tf.zeros([1]), name="bias")
            matmul = tf.squeeze(tf.sparse_tensor_dense_matmul(counts, W), axis=1)
            logits = tf.add(matmul, b)
        with tf.name_scope("loss"):
            regularizer = tf.nn.l2_loss(W)

        with tf.name_scope("prediction"):
            labels = tf.round(tf.sigmoid(logits))

        return labels, logits, regularizer

    def calculate_loss(self, logits, labels, regularizer, beta=0.01, pos_weight=1):
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                with tf.name_scope("loss"):
                    cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
                        logits=logits, targets=tf.to_float(labels), pos_weight=pos_weight)
                    cross_entropy_mean = tf.reduce_mean(cross_entropy, name="x_entropy_mean")

                    loss = tf.reduce_mean(cross_entropy_mean + beta * regularizer)

                    tf.summary.scalar("x_entropy_mean", cross_entropy_mean)
                    tf.summary.scalar("loss", loss)
                    return loss

    def make_train_op(self, loss, rate, epsilon, initialize=True):
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                with tf.name_scope("train"):
                    optimizer = tf.train.AdamOptimizer(learning_rate=rate, epsilon=epsilon)
                    op = optimizer.minimize(loss)

                    if initialize:
                        self.session.run(tf.global_variables_initializer())

                    return op

    def train(self, counts, labels, epochs=1, batch_size=32, rate=0.0001, epsilon=1e-8, beta=0.01, pos_weight=10, reset=True):
        if reset:
            self.reset()
            loss = self.calculate_loss(self.logits, self.target_labels, self.regularizer, beta, pos_weight)
            # self.loss = loss
            self.train_op = self.make_train_op(loss, rate, epsilon, initialize=reset)

        indices = list(range(len(labels)))
        writer = tf.summary.FileWriter(self.summary_path, self.graph)

        print("Training")

        for _ in range(epochs):
            self.epoch += 1
            print("===============")
            print("EPOCH", self.epoch)
            print("===============")

            random.shuffle(indices)
            batches = chunks(indices, batch_size)
            i = 0

            for batch in batches:
                self.it += 1
                i += 1
                if i % 100 == 0:
                    print("batch", i)
                batch_counts = csrToStv(counts[batch])
                summary, _ = self.session.run([self.summary, self.train_op],
                                              {self.counts: batch_counts,
                                               self.target_labels: labels[batch]})
                writer.add_summary(summary, self.it)
        writer.close()

    def test(self, counts, labels):
        pred = np.empty_like(labels)
        print("Testing")
        for index in range(len(labels)):
            pred[index] = self.session.run(self.pred_labels,
                                           {self.counts: csrToStv(counts[index]),
                                            self.target_labels: [labels[index]]})[0]
        print("AUC:", roc_auc_score(labels, pred))
        print("Precision:", precision_score(labels, pred))
        print("Recall:", recall_score(labels, pred))
        print("F1:", f1_score(labels, pred))
        return pred

    def train_test(self, train_counts, train_labels, test_counts, test_labels, epochs=1, *args, **kwargs):
        self.train(train_counts, train_labels, *args, **kwargs)
        preds = []
        preds.append(self.test(test_counts, test_labels))
        for _ in range(1, epochs):
            self.train(train_counts, train_labels, *args, **kwargs, reset=False)
            preds.append(self.test(test_counts, test_labels))
        return preds


    def save(self, pathname=None):
        if pathname is None:
            pathname = self.summary_path
        self.saver.save(self.session, pathname+"/model.ckpt")

    def restore(self, run_num=None, pathname=None):
        if pathname is None:
            if run_num is None:
                run_num = self.run_num - 1
            pathname = self.base_summary_path.format(run_num)
            if not os.path.exists(pathname):
                pathname = glob.glob(pathname + "-*")[0]

        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.saver.restore(self.session, pathname+"/model.ckpt")

def csrToStv(csr):
    indices = np.array(csr.nonzero()).T
    return tf.SparseTensorValue(indices, csr.data, csr.shape)

def toTfidf(train, dev, test):
    data = train + dev + test
    comments = [c.comment for c in data]
    sparse_counts = TfidfVectorizer(norm='l2', ngram_range=(1,5)).fit_transform(comments)  # lowercase = False?

    train_counts = sparse_counts[:len(train)]
    dev_counts = sparse_counts[len(train):len(train) + len(dev)]
    test_counts = sparse_counts[len(train) + len(dev):]

    train_labels = np.array([c.average.attack > 0.5 for c in train])
    dev_labels = np.array([c.average.attack > 0.5 for c in dev])
    test_labels = np.array([c.average.attack > 0.5 for c in test])

    return train_counts, train_labels, dev_counts, dev_labels, test_counts, test_labels

def save_tfidf(train_counts, train_labels, dev_counts, dev_labels, test_counts, test_labels, path=""):
    save_csr(os.path.join(path, "train_counts"), train_counts)
    np.save(os.path.join(path, "train_labels"), train_labels)
    save_csr(os.path.join(path, "dev_counts"), dev_counts)
    np.save(os.path.join(path, "dev_labels"), dev_labels)
    save_csr(os.path.join(path, "test_counts"), test_counts)
    np.save(os.path.join(path, "test_labels"), test_labels)

def load_tfidf(path=""):
    return load_csr(os.path.join(path, "train_counts.npz")),\
           np.load(os.path.join(path, "train_labels.npy")),\
           load_csr(os.path.join(path, "dev_counts.npz")),\
           np.load(os.path.join(path, "dev_labels.npy")),\
           load_csr(os.path.join(path, "test_counts.npz")),\
           np.load(os.path.join(path, "test_labels.npy"))

def save_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr,
             shape=array.shape)

def load_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])
