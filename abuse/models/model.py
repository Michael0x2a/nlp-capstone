from typing import Dict, List, Any, Generic, TypeVar, Iterable 

import sklearn.metrics as metrics

import utils.file_manip as fmanip

TSelf = TypeVar('TSelf', bound='Model')
TInput = TypeVar('TInput')

class ClassificationMetrics:
    def __init__(self, y_expected: List[List[bool]], 
                       y_predicted: List[List[bool]]) -> None:
        self.accuracy = metrics.accuracy_score(y_expected, y_predicted)
        self.precision = metrics.precision_score(y_expected, y_predicted)
        self.recall = metrics.recall_score(y_expected, y_predicted)
        self.f1 = metrics.f1_score(y_expected, y_predicted)
        self.roc_auc = metrics.roc_auc_score(y_expected, y_predicted)

        self.confusion_matrix = metrics.confusion_matrix(y_expected, y_predicted)

    def to_table_row(self) -> str:
        return "| {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                self.accuracy,
                self.precision,
                self.recall,
                self.f1,
                self.roc_auc)

    def get_header(self) -> str:
        return ("| Accuracy | Precision | Recall | F1 | ROC |\n" +
                "| -------- | --------- | ------ | -- | --- |")


class Model(Generic[TInput]):
    # Core methods that must be implemented
    def __init__(self, **params) -> None:
        '''For the sake of consistency, the constructor for all subclasses
        should take in an arbitrary set of parameters, and do very little
        else. All parameters should have a default value.'''
        raise NotImplementedError()

    def get_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def _save_model(self, path: str) -> None:
        raise NotImplementedError()

    def _restore_model(self, path: str) -> None:
        raise NotImplementedError()

    def build_model(self) -> None:
        '''Builds and saves the model, using the currently set params.'''
        raise NotImplementedError()

    def train(self, xs: List[TInput], ys: List[List[bool]]) -> None:
        '''Trains the model. The expectation is that this method is called
        exactly once.'''
        raise NotImplementedError()

    def predict(self, xs: List[TInput]) -> List[List[bool]]:
        raise NotImplementedError()

    # Useful utility methods
    def predict_single(self, x: TInput) -> List[bool]:
        return self.predict([x])[0]

    @classmethod
    def restore_from_saved(cls: Any, path: str) -> Any:
        # Signature really should be
        # (Type[TSelf], str) -> TSelf
        # ...but idk if mypy supports this fully atm
        obj = cls(**fmanip.load_json(fmanip.join(path, 'params.json')))
        obj._restore_model(fmanip.join(path, 'model'))
        return obj

    def save(self, path: str) -> None:
        fmanip.ensure_folder_exists(path)
        param_path = fmanip.join(path, 'params.json')
        model_path = fmanip.join(path, 'model')
        fmanip.ensure_folder_exists(model_path)

        fmanip.write_nice_json(self.get_parameters(), param_path)
        self._save_model(model_path)

