from typing import List, Tuple, Optional, Dict, Generator, TypeVar
import os
import glob
import random
import ast
from math import ceil
import numpy as np  # type: ignore
from scipy.sparse import csr_matrix, spdiags  # type: ignore
import tensorflow as tf  # type: ignore
from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, roc_curve  # type: ignore
from scipy.stats import spearmanr  # type: ignore

from custom_types import *
from models.model import Model

# default path to save things in
SUMMARY_PATH = "runs/lr{0}"

T = TypeVar("T")

def chunks(l: List[T], n: int) -> Generator[List[T], None, None]:
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

class LogisticClassifier(Model[str]):
    def __init__(self,
                 num_grams: int=30000,
                 batch_size: int=32,
                 epochs: int=20,
                 pos_weight: float=10,
                 lambda_: float=1e-7,
                 learning_rate: float=0.0001,
                 beta1: float=0.9,
                 beta2: float=0.999,
                 epsilon: float=1e-8,
                 base_summary_path: str=SUMMARY_PATH,
                 restore_from: Optional[str]=None,
                 run_num: int=0,
                 it: int=0,
                 epoch: int=0,
                 vocabulary: Optional[str]=None,
                 idf: Optional[List[float]]=None) -> None:
        self.num_grams = num_grams
        self.batch_size = batch_size
        self.epochs = epochs
        self.pos_weight = pos_weight
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.base_summary_path = base_summary_path

        self.run_num = run_num
        self.it = it
        self.epoch = epoch
        if vocabulary is None or idf is None:
            self.vocabulary = None
            self.idf = None
        else:
            self.vocabulary = ast.literal_eval(vocabulary)
            self.idf = np.array(idf)

        self.config = tf.ConfigProto(allow_soft_placement=True)
        self.graph = tf.Graph()

        self._build()

        if restore_from is None:
            self.reset()
        else:
            self._restore_model(restore_from)


    def reset(self) -> None:
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.session = tf.Session(graph=self.graph, config=self.config)
                self.session.run(self.init)

        self.run_num = self._get_next_run_num()
        self.it = 0
        self.epoch = 0

        if self.vocabulary is None or self.idf is None:
            self.tfidf_vectorizer = None

    def _get_next_run_num(self) -> int:
        i = 0
        while True:
            i += 1
            summary_path = self.base_summary_path.format(i)
            if not os.path.exists(summary_path) and not glob.glob(summary_path + "-*"):
                return i

    def _get_summary_path(self) -> str:
        return self.base_summary_path.format(self.run_num)

    def get_parameters(self) -> Dict[str, Any]:
        return {
                "num_grams": self.num_grams,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "pos_weight": self.pos_weight,
                "lambda_": self.lambda_,
                "learning_rate": self.learning_rate,
                "beta1": self.beta1,
                "beta2": self.beta2,
                "epsilon": self.epsilon,
                "base_summary_path": self.base_summary_path,
                "run_num": self.run_num,
                "it": self.it,
                "epoch": self.epoch,
                "vocabulary": str(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer is not None else None,
                "idf": self.tfidf_vectorizer.idf_.tolist() if self.tfidf_vectorizer is not None else None
        }

    def _build(self) -> None:
        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self._build_inputs()
                self._build_model()
                self._build_loss()
                self._build_optimizer()
                self.init = tf.global_variables_initializer()
                self.summary = tf.summary.merge_all()
                self.saver = tf.train.Saver()

    def _build_inputs(self) -> None:
        if self.vocabulary is not None and self.idf is not None:
            self.tfidf_vectorizer = TfidfVectorizer(
                norm="l2",
                ngram_range=(1,5),
                max_features=self.num_grams,
                vocabulary=self.vocabulary,
                dtype=np.float32)
            self.tfidf_vectorizer._tfidf._idf_diag = spdiags(self.idf, 0, len(self.idf), len(self.idf))

        with tf.name_scope("input"):
            self.counts = tf.sparse_placeholder(tf.float32, [None, self.num_grams], "counts")
            self.target_labels = tf.placeholder(tf.bool, [None], "target_labels")

    def _build_model(self) -> None:
        with tf.name_scope("model"):
            W = tf.Variable(tf.random_normal([self.num_grams, 1]), name="weight")
            b = tf.Variable(tf.zeros([1]), name="bias")
            matmul = tf.squeeze(tf.sparse_tensor_dense_matmul(self.counts, W), axis=1)
            self.logits = tf.add(matmul, b)

        with tf.name_scope("regularizer"):
            self.regularizer = tf.nn.l2_loss(W, name="regularizer")

        with tf.name_scope("probabilities"):
            self.probabilities = tf.sigmoid(self.logits)

    def _build_loss(self) -> None:
        with tf.name_scope("loss"):
            cross_entropy = tf.nn.weighted_cross_entropy_with_logits(
                logits=self.logits, targets=tf.to_float(self.target_labels), pos_weight=self.pos_weight)
            cross_entropy_mean = tf.reduce_mean(cross_entropy, name="x_entropy_mean")
            self.loss = tf.reduce_mean(cross_entropy_mean + self.lambda_ * self.regularizer)

            tf.summary.scalar("x_entropy_mean", cross_entropy_mean)
            tf.summary.scalar("loss", self.loss)

    def _build_optimizer(self) -> None:
        with tf.name_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate,
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon).minimize(self.loss)

    def train(self, comments: List[str], labels: List[int], epochs: Optional[int]=None, batch_size: Optional[int]=None, **params: Any) -> None:
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(
                norm="l2",
                ngram_range=(1,5),
                max_features=self.num_grams,
                dtype=np.float32)  # lowercase = False?
            self.tfidf_vectorizer.fit(comments)  # type: ignore
        counts = self.tfidf_vectorizer.transform(comments)  # type: ignore
        labels = np.array(labels, dtype=np.bool)  # type: ignore
        self._train(counts, labels, epochs, batch_size)  # type: ignore

    def _train(self, counts: List[List[float]], labels: List[bool], epochs: Optional[int]=None, batch_size: Optional[int]=None) -> None:
        if epochs is None:
            epochs = self.epochs
        if batch_size is None:
            batch_size = self.batch_size

        batch_count = ceil(len(labels) / batch_size)
        indices = list(range(len(labels)))
        writer = tf.summary.FileWriter(self._get_summary_path(), self.graph)

        for _ in range(epochs):
            self.epoch += 1
            print("=== EPOCH {} ===".format(self.epoch))

            random.shuffle(indices)
            batches = chunks(indices, batch_size)
            i = 0

            for batch in batches:
                self.it += 1
                i += 1
                if i % 10 == 0:
                    print("Training batch", i, "of", batch_count, end="\r")
                batch_counts = csr_to_stv(counts[batch])  # type: ignore
                summary, _ = self.session.run([self.summary, self.optimizer],
                                              {self.counts: batch_counts,
                                               self.target_labels: labels[batch]})  # type: ignore
                writer.add_summary(summary, self.it)
            # clear current line for any reasonable batch count
            print("                                      ", end="\r")
        writer.close()

    def predict(self, comments: List[str]) -> List[List[float]]:
        counts = self.tfidf_vectorizer.transform(comments)  # type: ignore
        return self._predict(counts)

    def _predict(self, counts: List[List[float]]) -> List[List[float]]:
        num_comments = counts.shape[0]  # type: ignore
        probs = np.empty(num_comments, dtype=np.float32)
        for i, count in enumerate(counts):
            if i % 500 == 0:
                print("Testing number", i, "of", num_comments, end="\r")
            probs[i] = self.session.run(self.probabilities,
                                        {self.counts: csr_to_stv(count)})[0]
        print("                                      ", end="\r")
        return np.array([1-probs, probs]).T  # type: ignore
    
    def _test(self, counts: List[List[float]], labels: List[bool]) -> List[List[float]]:
        probs = self._predict(counts)
        pred = probs.argmax(axis=0)  # type: ignore
        evaluate(labels, pred)
        return probs

    def _train_test(self, train_counts: List[List[float]], train_labels: List[bool], test_counts: List[List[float]], test_labels: List[bool],
                   epochs: Optional[int]=None, batch_size: Optional[int]=None, test_interval: int=2) -> List[List[List[float]]]:
        if epochs is None:
            epochs = self.epochs
        self._train(train_counts, train_labels, epochs=1, batch_size=batch_size)
        preds = []
        if test_interval == 1:
            preds.append(self._test(test_counts, test_labels))
        for i in range(1, epochs):
            self._train(train_counts, train_labels, epochs=1, batch_size=batch_size)
            if (i+1) % test_interval == 0:
                preds.append(self._test(test_counts, test_labels))
        return preds

    def _save_model(self, path: str) -> None:
        self.save_(pathname=path)

    def save_(self, filename: str="model.ckpt", pathname: Optional[str]=None) -> None:
        if pathname is None:
            pathname = self._get_summary_path()
        self.saver.save(self.session, os.path.join(pathname, filename))  # type: ignore

    def _restore_model(self, path: str) -> None:
        self.restore_(pathname=path)

    def restore_(self, run_num: Optional[int]=None, filename: str="model.ckpt", pathname: Optional[str]=None) -> None:
        # doesn't restore summary_path, run_num, it count, or epoch
        if pathname is None:
            if run_num is None:
                run_num = self.run_num - 1
            pathname = self.base_summary_path.format(run_num)
            if not os.path.exists(pathname):
                run_paths = glob.glob(pathname + "-*")
                if len(run_paths) < 1:
                    print("No run with that number.")
                    return
                elif len(run_paths) > 1:
                    print("Multiple runs with that number.")
                pathname = run_paths[0]

        with self.graph.as_default():
            with self.graph.device("/gpu:0"):
                self.session = tf.Session(graph=self.graph, config=self.config)
                self.saver.restore(self.session, os.path.join(pathname, filename))  # type: ignore

def evaluate(labels: List[bool], pred: List[bool]) -> None:
    print("AUC:", roc_auc_score(labels, pred))
    print("Spearman:", spearmanr(labels, pred).correlation)
    print("Precision:", precision_score(labels, pred))
    print("Recall:", recall_score(labels, pred))
    print("F1:", f1_score(labels, pred))

def csr_to_stv(csr: List[T]) -> List[T]:
    indices = np.array(csr.nonzero()).T  # type: ignore
    return tf.SparseTensorValue(indices, csr.data, csr.shape)  # type: ignore

def to_tfidf(train, dev, test):  # type: ignore
    data = train + dev + test
    comments = [c.comment for c in data]
    sparse_counts = TfidfVectorizer(norm="l2", ngram_range=(1,5), max_features=30000, dtype=np.float32).fit_transform(comments)  # lowercase = False?

    train_counts = sparse_counts[:len(train)]
    dev_counts = sparse_counts[len(train):len(train) + len(dev)]
    test_counts = sparse_counts[len(train) + len(dev):]

    train_labels = np.array([c.average.attack > 0.5 for c in train])
    dev_labels = np.array([c.average.attack > 0.5 for c in dev])
    test_labels = np.array([c.average.attack > 0.5 for c in test])

    return train_counts, train_labels, dev_counts, dev_labels, test_counts, test_labels

def save_tfidf(train_counts, train_labels, dev_counts, dev_labels, test_counts, test_labels, path=""):  # type: ignore
    save_csr(os.path.join(path, "train_counts"), train_counts)
    np.save(os.path.join(path, "train_labels"), train_labels)
    save_csr(os.path.join(path, "dev_counts"), dev_counts)
    np.save(os.path.join(path, "dev_labels"), dev_labels)
    save_csr(os.path.join(path, "test_counts"), test_counts)
    np.save(os.path.join(path, "test_labels"), test_labels)

def load_tfidf(path=""):  # type: ignore
    return load_csr(os.path.join(path, "train_counts.npz")),\
           np.load(os.path.join(path, "train_labels.npy")),\
           load_csr(os.path.join(path, "dev_counts.npz")),\
           np.load(os.path.join(path, "dev_labels.npy")),\
           load_csr(os.path.join(path, "test_counts.npz")),\
           np.load(os.path.join(path, "test_labels.npy"))

def save_csr(filename: str, array: List[T]) -> None:  # type: ignore
    np.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr,  # type: ignore
             shape=array.shape)  # type: ignore

def load_csr(filename: str) -> List[List[float]]:  # type: ignore
    loader = np.load(filename)
    return csr_matrix((loader["data"], loader["indices"], loader["indptr"]),
                      shape=loader["shape"])

