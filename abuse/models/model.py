from typing import Dict, List, Any, Generic, TypeVar, Iterable, Optional, Tuple
import os.path
import glob

import sklearn.metrics as metrics  # type: ignore
import scipy.stats as stats  # type: ignore
import numpy as np  # type: ignore
import matplotlib.pyplot as plt  # type: ignore

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
        with open(os.path.join(path, "false_positives.txt"), 'w') as fp, \
             open(os.path.join(path, "false_negatives.txt"), 'w') as fn, \
             open(os.path.join(path, "true_positives.txt"), 'w') as tp, \
             open(os.path.join(path, "true_negatives.txt"), 'w') as tn:
            for comment, exp, pred in zip(self.x, self.y_expected, self.y_predicted):
                f = None
                
                true_bad = exp > self.threshold
                pred_bad = pred > self.threshold

                if true_bad and not pred_bad:
                    f = fn
                elif not true_bad and pred_bad:
                    f = fp
                elif true_bad and pred_bad:
                    f = tp
                elif not true_bad and not pred_bad:
                    f = tn
                else:
                    raise AssertionError()

                if f is not None:
                    f.write("{:.6f} {:.6f} {}\n".format(exp, pred, comment.encode("utf-8")))

def one_hot(y: List[int]) -> List[List[int]]:
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

        self.accuracy = metrics.accuracy_score(y_expected, y_predicted)
        self.precision = metrics.precision_score(y_expected, y_predicted, average='binary')
        self.recall = metrics.recall_score(y_expected, y_predicted, average='binary')
        self.f1 = metrics.f1_score(y_expected, y_predicted, average='binary')
        self.roc_auc = metrics.roc_auc_score(y_expected_hot, y_predicted_prob, average='binary')
        self.spearman = stats.spearmanr(y_expected, y_predicted).correlation
        self.confusion_matrix = metrics.confusion_matrix(y_expected, y_predicted)
        self.fpr, self.tpr, self.thr = metrics.roc_curve(
                y_expected, 
                y_predicted_prob[:,1])

    def to_table_row(self) -> str:
        return "| {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
                self.accuracy,
                self.precision,
                self.recall,
                self.f1,
                self.roc_auc,
                self.spearman)

    def get_header(self) -> str:
        return ("| Accuracy | Precision | Recall | F1 | ROC | Spearman |\n" +
                "| -------- | --------- | ------ | -- | --- | -------- |")

    def make_roc_curve(self, save_path: str=None) -> Tuple[List[float], List[float], List[float]]:
        plt.plot(self.fpr, self.tpr)
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        return self.fpr, self.tpr, self.thr


class Model(Generic[TInput]):
    # default log dir; override this
    base_log_dir = "runs/run{}"

    def __init__(self,
                 restore_from: Optional[str] = None,
                 run_num: Optional[int]=None,
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
        self.run_num = self._get_next_run_num() if run_num is None else run_num
        if restore_from is not None:
            self._restore_model(restore_from)

    # Core methods that must be implemented

    def _get_parameters(self) -> Dict[str, Any]:
        '''Returns all parameters for this class. This will be used
        when saving/loading models.'''
        raise NotImplementedError()

    def _save_model(self, path: str) -> None:
        '''Saves the model. The path is a path to an existing folder;
        this method may create any arbitrary files/folders within the
        provided path.'''
        raise NotImplementedError()

    def _restore_model(self, path: str) -> None:
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
    def _get_next_run_num(cls: Any) -> int:
        i = 0
        while True:
            i += 1
            log_dir = cls.base_log_dir.format(i)
            if not os.path.exists(log_dir) and not glob.glob(log_dir + "-*"):
                return i

    def _get_log_dir(self) -> str:
        return self.base_log_dir.format(self.run_num)

    def format_log_dir(self, path:Optional[str]) -> str:
        return self._get_log_dir() if path is None\
                                   else path.format(self._get_log_dir())

    def _get_all_parameters(self) -> Dict[str, Any]:
        base_params = { "run_num": self.run_num }
        return { **base_params, **self._get_parameters() }

    @classmethod
    def restore_from_saved(cls: Any,
                           run_num: Optional[int]=None,
                           path: Optional[str]=None) -> Any:
        # Signature really should be
        # (Type[TSelf], str) -> TSelf
        # ...but idk if mypy supports this fully atm
        '''Restores model and parameters from given location
        If run num is passed, tries to find that run's path using the base log dir;
        if no run num is passed, uses the last run's path. If path is passed, formats
        the given string with the run path; else just restores from the run path.
        (E.g. path="{}/epoch10", run_num=4 -> "runs/run4/epoch10")'''
        print(run_num, path)
        if run_num is None:
            run_num = cls._get_next_run_num() - 1
        run_dir = cls.base_log_dir.format(run_num)
        if not os.path.exists(run_dir):
            run_dirs = glob.glob(run_dir + "-*")
            if len(run_dirs) < 1:
                print("Error: No run with that number.")
                return
            elif len(run_dirs) > 1:
                print("Multiple runs with that number.")
            run_dir = run_dirs[0]

        path = run_dir if path is None else path.format(run_dir)

        assert os.path.isdir(path)
        return cls(
                restore_from=path,
                **fmanip.load_json(fmanip.join(path, 'params.json')))

    def save(self, path: Optional[str]=None) -> None:
        '''Saves the model and parameters. The path can be a string to be
        formatted with the default path (including a completely different
        path that won't be formatted) or None to use the default path of
        the log dir. (E.g. path="{}/epoch100" -> "runs/run10/epoch100")'''
        path = self.format_log_dir(path)
        fmanip.ensure_folder_exists(path)
        param_path = fmanip.join(path, 'params.json')
        fmanip.write_nice_json(self._get_all_parameters(), param_path)
        self._save_model(path)

