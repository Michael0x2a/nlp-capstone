from typing import Optional, Dict, Tuple, cast, Any

from sklearn.pipeline import Pipeline  # type: ignore
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.externals import joblib  # type: ignore
import sklearn.metrics  # type: ignore
import numpy as np  # type: ignore

from custom_types import *
from data_extraction.wikipedia import *
from models.model import Model 
import utils.file_manip as fmanip


class BagOfWordsClassifier(Model[str]):
    base_log_dir = "runs/bow/run{}"

    '''
    Bag of words classifier:

    --max_features [int; default = 10000]
        Max number of features to use in count vectorizer

    TODO: Finish documenting
    '''

    # Core methods that must be implemented
    def __init__(self, restore_from: Optional[str] = None,
                       run_num: Optional[int]=None,
                       max_features: int = 10000,
                       ngram_range_lo: int = 1,
                       ngram_range_hi: int = 2,
                       norm: str = 'l2') -> None:
        self.max_features = max_features
        self.ngram_range_lo = ngram_range_lo
        self.ngram_range_hi = ngram_range_hi
        self.norm = norm

        if restore_from is None:
            self.classifier = Pipeline([
                ('vect', CountVectorizer(
                            max_features=10000, 
                            ngram_range=(self.ngram_range_lo, self.ngram_range_hi))),
                ('tfidf', TfidfTransformer(norm='l2')),
                ('clf', LogisticRegression())
            ])
        super().__init__(restore_from, run_num)

    def _restore_model(self, path: str) -> None:
        self.classifier = joblib.load(fmanip.join(path, 'classifier.pkl'))

    def _get_parameters(self) -> Dict[str, Any]:
        return {
                'max_features': self.max_features,
                'ngram_range_lo': self.ngram_range_lo,
                'ngram_range_hi': self.ngram_range_hi,
                'norm': self.norm,
        }

    def _save_model(self, path: str) -> None:
        # lol, apparently pickling is the recommended way of saving/loading
        # trained classifiers. See 
        # http://scikit-learn.org/stable/modules/model_persistence.html
        #
        # The pickled output is relatively fragile, and could break on
        # different operating systems/different version of python/different
        # versions of basically any library we're using.
        joblib.dump(self.classifier, fmanip.join(path, 'classifier.pkl'))

    def train(self, xs: List[str], ys: List[int], **params: Any) -> None:
        '''Trains the model. The expectation is that this method is called
        exactly once.'''
        if len(params) != 0:
            raise Exception("Bag of words does not take in any extra params to train")
        self.classifier.fit(xs, ys)

    def report_contents(self) -> None:
        vec = self.classifier.steps[0][1]
        reg = self.classifier.steps[2][1]

        uni_words = [word for word in vec.vocabulary_ if ' ' not in word]
        bi_words = [word for word in vec.vocabulary_ if ' ' in word]
        with open('interesting_data/bag_of_words_unigram_toxicity_results.txt', 'w') as stream:
            out = [(word, self.classifier.predict_proba([word])) for word in uni_words]

            target_class = 1
            out.sort(key=lambda x: x[1][0][target_class])
            for x in out[:1000]:
                stream.write(str(x) + '\n')
            stream.write('\n')
            for x in out[-1000:]:
                stream.write(str(x) + '\n')
        with open('interesting_data/bag_of_words_bigram_toxicity_results.txt', 'w') as stream:
            out = [(word, self.classifier.predict_proba([word])) for word in bi_words]

            target_class = 1
            out.sort(key=lambda x: x[1][0][target_class])
            for x in out[:1000]:
                stream.write(str(x) + '\n')
            stream.write('\n')
            for x in out[-1000:]:
                stream.write(str(x) + '\n')

    def predict(self, xs: List[str]) -> List[List[float]]:
        return cast(List[List[float]], self.classifier.predict_proba(xs))

    def predict_log(self, xs: List[str]) -> List[List[float]]:
        return cast(List[List[float]], self.classifier.predict_log_proba(xs))

