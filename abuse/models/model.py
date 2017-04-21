from typing import Dict, List, Any, Generic, TypeVar, Iterable

import sklearn.metrics as metrics

import utils.file_manip as fmanip

TSelf = TypeVar('TSelf', bound='Model')
TInput = TypeVar('TInput')
TLabel = TypeVar('TLabel')

class ClassificationMetrics(Generic[TLabel]):
    def __init__(self, y_expected: List[TLabel], 
                       y_predicted: List[TLabel]) -> None:
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
                "| -------- | --------- | ------ | -- | --- |\n")


class Model(Generic[TInput, TLabel]):
    # Core methods that must be implemented
    def __init__(self) -> None:
        '''For the sake of consistency, the constructor for all subclasses
        should take in no params and only set default values for parameters.'''
        raise NotImplementedError()

    def build_model(self) -> None:
        '''Builds and saves the model, using the currently set params.'''
        raise NotImplementedError()

    def reset(self) -> None:
        '''Resets the model, clearing all training data/any learned information'''
        raise NotImplementedError()

    def train(self, xs: List[TInput], ys: List[TLabel], **params) -> None:
        '''Trains the model. The expectation is that this method is called
        exactly once.'''
        raise NotImplementedError()

    def train_iterative(self, xs: List[TInput], ys: List[TLabel], **params) -> Iterable[List[TLabel]]:
        '''Also trains the model, but should yield after each epoch so the client
        can analyze data/collect statistics.'''
        raise NotImplementedError()

    def predict_single(self, xs: TInput) -> TLabel:
        raise NotImplementedError()

    def predict(self, xs: List[TInput]) -> List[TLabel]:
        return [self.evaluate_single(x) for x in xs]

    def _save_model(self, path: str) -> None:
        raise NotImplementedError()

    def _restore_model(self, path: str) -> None:
        raise NotImplementedError()

    def get_parameters(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def set_parameters(self, params: Dict[str, Any]) -> None:
        raise NotImplementedError()

    # Useful utility methods

    @classmethod
    def restore_from_saved(cls, path: str) -> None:
        self.set_parameters(fmanip.load_json(fmanip.join(path, 'params.json')))
        self._restore_model(fmanip.join(path, 'model'))

    @classmethod
    def make_new(cls: TSelf, **params: Any) -> TSelf:
        model = cls()
        model.set_parameters(params)
        model.build_model()
        return model

    def save(self, path: str) -> None:
        fmanip.ensure_folder_exists(path)
        param_path = fmanip.join(path, 'params.json')
        model_path = fmanip.ensure_folder_exists(fmanip.join(path, 'model'))

        fmanip.write_nice_json(self.get_parameters(), param_path)
        self._save_model(model_path)

