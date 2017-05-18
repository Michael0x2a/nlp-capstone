from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


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


class CopiedClassifier(Model[str]):
    base_log_dir = "runs/lrcopy/run{}"

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
                ('vect', CountVectorizer()),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression()),
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
        params = {
            'vect__max_features': 10000,
            'vect__ngram_range': (1,5),
            'vect__analyzer' : 'char',
            'tfidf__sublinear_tf' : True,
            'tfidf__norm' :'l2',
            'clf__C' : 10,
        }
        self.classifier.set_params(**params).fit(xs, ys)

    def report(self) -> None:
        vec = self.classifier.steps[0][1]
        reg = self.classifier.steps[2][1]

        uni_char = [char for char in vec.vocabulary_ if len(char) == 1]
        with open('interesting_data/char_attack_uni.txt', 'w') as stream:
            out = [(c, self.classifier.predict_proba([c])) for c in uni_char]

            target_class = 1
            out.sort(key=lambda x: x[1][0][target_class])
            for x in out:
                stream.write(str(x) + '\n')
                print(x)

    def predict(self, xs: List[str]) -> List[List[float]]:
        return cast(List[List[float]], self.classifier.predict_proba(xs))

