from typing import Optional, Dict, Tuple, cast

from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
import sklearn.metrics  # type: ignore
import numpy as np  # type: ignore

from custom_types import *
from data_extraction.wikipedia import *
from models.model import Model, ClassificationMetrics



class BagOfWordsClassifier(Model[str]):
    '''
    Bag of words classifier:

    --max_features [int; default = 10000]
        Max number of features to use in count vectorizer

    TODO: Finish documenting
    '''

    # Core methods that must be implemented
    def __init__(self, restore_from: Optional[str] = None,
                       max_features: int = 10000,
                       ngram_range_lo: int = 1,
                       ngram_range_hi: int = 2,
                       norm: str = 'l2') -> None:
        self.max_features = max_features
        self.ngram_range_lo = ngram_range_lo
        self.ngram_range_hi = ngram_range_hi
        self.norm = norm

        self.classifier = Pipeline([
            ('vect', CountVectorizer(
                        max_features=10000, 
                        ngram_range=(self.ngram_range_lo, self.ngram_range_hi))),
            ('tfidf', TfidfTransformer(norm='l2')),
            ('clf', LogisticRegression())
        ])

    def get_parameters(self) -> Dict[str, Any]:
        return {
                'max_features': self.max_features,
                'ngram_range_lo': self.ngram_range_lo,
                'ngram_range_hi': self.ngram_range_hi,
                'norm': self.norm,
        }

    def _save_model(self, path: str) -> None:
        raise NotImplementedError()

    def train(self, xs: List[str], ys: List[int]) -> None:
        '''Trains the model. The expectation is that this method is called
        exactly once.'''
        self.classifier.fit(xs, ys)

    def predict(self, xs: List[str]) -> List[int]:
        return cast(List[int], self.classifier.predict(xs))


# TODO: Refactor; this is identical to the function in models/rnn_classifier.py
IS_ATTACK = 1
IS_OK = 0

def extract_data(comments: AttackData) -> Tuple[List[str], List[int]]:
    x_values = []
    y_values = []
    for comment in comments:
        x_values.append(comment.comment)
        cls = IS_ATTACK if comment.average.attack > 0.5 else IS_OK
        y_values.append(cls)
    return x_values, y_values


def main2() -> None:
    print("Starting...")

    # Classifier setup
    print("Building model...")
    classifier = BagOfWordsClassifier()

    # Load data
    print("Loading data...")
    train_data, dev_data, test_data = load_attack_data()
    x_train_raw, y_train = extract_data(train_data)
    x_dev_raw, y_dev = extract_data(dev_data)

    # Begin training
    print("Training...")
    classifier.train(x_train_raw, y_train)

    # Evaluation
    print("Evaluation...")
    y_predicted = classifier.predict(x_dev_raw)
    metrics = ClassificationMetrics(y_dev, y_predicted)

    print(metrics.get_header())
    print(metrics.to_table_row())
    print(metrics.confusion_matrix)


if __name__ == '__main__':
    '''data_src_path = 'data/wikipedia-detox-data-v6'
    print("Loading...")
    train, dev, test = load_raw_data(data_src_path)
    print("Training...")
    bag_of_words_model(train, dev, test)'''
    main2()

