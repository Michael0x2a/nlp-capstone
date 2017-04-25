from typing import Dict, List, Any, Generic, TypeVar, Iterable, Optional, Tuple
import os.path

import sklearn.metrics as metrics  # type: ignore
import scipy.stats as stats  # type: ignore
import numpy as np  # type: ignore

import utils.file_manip as fmanip

TSelf = TypeVar('TSelf', bound='Model')
TInput = TypeVar('TInput')

class ErrorAnalysis:
    def __init__(self, x: List[str], y_expected: List[float], y_predicted: List[float],
                 threshold: float = 0.5) -> None:
        self.x = x
        self.y_expected = y_expected
        self.y_predicted = y_predicted
        self.threshold = threshold

    def save_errors(self, path: str = "") -> None:
        with open(os.path.join(path, "false_positives.txt") as fp,
             open(os.path.join(path, "false_negatives.txt") as np:
            for comment, exp, pred in zip(self.x, self.y_expected, self.y_predicted):
                f = None
                if exp > self.threshold and pred < self.threshold:
                    f = fn
                elif exp < self.threshold and pred > self.threshold:
                    f = fp
                if f is not None:
                    f.write("{:.6f} {:.6f} {}\n".format(exp, pred, comment))

def one_hot(y):
    out = []
    for i in y:
        if i == 0:
            out.append([1, 0])
        else:
            out.append([0, 1])
    return out

class BinaryClassificationMetrics:
    def __init__(self, y_expected: List[int], 
                       y_predicted_prob: List[List[float]]) -> None:
        y_expected_hot = np.array(one_hot(y_expected))
        y_predicted = np.argmax(y_predicted_prob, 1)

        print(y_expected_hot)
        print(y_predicted)

        self.accuracy = metrics.accuracy_score(y_expected, y_predicted)
        self.precision = metrics.precision_score(y_expected, y_predicted)
        self.recall = metrics.recall_score(y_expected, y_predicted)
        self.f1 = metrics.f1_score(y_expected, y_predicted)
        self.roc_auc = metrics.roc_auc_score(y_expected_hot, y_predicted_prob, average='macro')
        self.spearman = stats.spearmanr(y_expected, y_predicted).correlation
        self.confusion_matrix = metrics.confusion_matrix(y_expected, y_predicted)

    def to_table_row(self) -> str:
        return "| {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                self.accuracy,
                self.precision,
                self.recall,
                self.f1,
                self.roc_auc,
                self.spearman)

    def get_header(self) -> str:
        return ("| Accuracy | Precision | Recall | F1 | ROC | Spearman |\n" +
                "| -------- | --------- | ------ | -- | --- | -------- |")


class Model(Generic[TInput]):
    # Core methods that must be implemented
    def __init__(self, restore_from: Optional[str] = None, 
                       **params: Any) -> None:
        '''The constructor is responsible for storing all
        relevant parameters, and building the model. All parameters
        must have a default value, if not specified.
        
        If the 'restore_from' path is provided, the constructor should:

        1. Load the model/any saved or trained results from that path
           instead of building the model from scratch.
        2. Assume that the provided params correspond to the exact
           parameters set when the model was saved. (So, you don't need
           to overwrite the params at all).

        The 'restore_from' path will always be a path to an existing
        folder; each class can save/load arbitrary files to that folder.
        '''
        raise NotImplementedError()

    def get_parameters(self) -> Dict[str, Any]:
        '''Returns all parameters for this class. This will be used
        when saving/loading models.'''
        raise NotImplementedError()

    def _save_model(self, path: str) -> None:
        '''Saves the model. The path is a path to an existing folder;
        this method may create any arbitrary files/folders within the
        provided path.'''
        raise NotImplementedError()

    def train(self, xs: List[TInput], ys: List[int], **params: Any) -> None:
        '''Trains the model. The expectation is that this method is called
        exactly once. The model can also accept additional params to tweak
        the behavior of the training method in some way. Note that cmd.py
        will completely ignore the kwargs, so the 'train' method shouldn't
        rely on any of them being present.'''
        raise NotImplementedError()

    def predict(self, xs: List[TInput]) -> List[List[float]]:
        raise NotImplementedError()

    # Useful utility methods
    def predict_single(self, x: TInput) -> List[float]:
        return self.predict([x])[0]

    @classmethod
    def restore_from_saved(cls: Any, path: str) -> Any:
        # Signature really should be
        # (Type[TSelf], str) -> TSelf
        # ...but idk if mypy supports this fully atm
        assert os.path.isdir(path)
        return cls(
                restore_from=path, 
                **fmanip.load_json(fmanip.join(path, 'params.json')))

    def save(self, path: str) -> None:
        fmanip.ensure_folder_exists(path)
        param_path = fmanip.join(path, 'params.json')
        fmanip.write_nice_json(self.get_parameters(), param_path)
        self._save_model(path)

