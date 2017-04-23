'''Command line interface for extracting data. Note: I'm
not using any command line libraries and am just manually
munging sys.argv. This is more out of laziness more then
anything.'''

from typing import Tuple, Dict, Any, Union, List
import sys

from data_extraction.wikipedia import *
from models.bag_of_words import BagOfWordsClassifier
from models.model import Model, ClassificationMetrics

Primitive = Union[int, float, str, bool]
Data = Tuple[List[str], List[int]]

def get_wikipedia_data(category: str = None,
                       attribute: str = None,
                       threshold: float = 0.5, 
                       use_dev: bool = True,
                       use_small: bool = False) -> Tuple[Data, Data]:
    '''
    Personal attacks dataset:

    --category [str] 
        The wikipedia category to use. Should be either 'attack', 
        'toxicity', or 'aggressiveness'.

    --attribute [str]
        The particular attribute from that dataset to test on.
        Defaults to the category name if not set.

    --threshold [float; default = 0.5]
        Any comments with an average attribute score larger then the
        threshold is labeled as an '1' (and is otherwise 0).
                    
    --use_dev [bool; default = True]
        If true, uses the dev dataset for evaluation, otherwise 
        uses the test dataset.

    --use_small [bool; default = False]
        If true, uses the shorter version of the dataset.
    '''
    funcs = {
            'personal_attack': load_attack_data,
            'toxicity': load_toxicity_data,
            'aggression': load_aggression_data
    }

    assert category is not None and category in funcs
    if attribute is None:
        attribute = category
    assert attribute is not None

    def extract_data(comments: AttackData) -> Data:
        x_values = []
        y_values = []
        for comment in comments:
            x_values.append(comment.comment)
            cls = 1 if getattr(comment.average, attribute) > threshold else 0  # type: ignore
            y_values.append(cls)
        return x_values, y_values
    train_data, dev_data, test_data = funcs[category](small=use_small)  # type: ignore

    train = extract_data(train_data)
    if use_dev:
        test = extract_data(dev_data)
    else:
        test = extract_data(test_data)
        
    return train, test

def main() -> None:
    # List of registered datasets and models
    datasets = {
            'wikipedia': get_wikipedia_data,
    }

    models = {
            'bag_of_words': BagOfWordsClassifier,
    }

    # Extracting and verifying command line args
    info = parse_args(sys.argv[1:])
    verify_choice('Dataset', datasets, info.dataset_name)
    verify_choice('Model', models, info.model_name)

    dataset_func = datasets[info.dataset_name]
    model_class = models[info.model_name]

    verify_help(dataset_func, info.dataset_params)
    verify_help(model_class, info.model_params)

    
    # Ok, go
    print("Loading {} data...".format(info.dataset_name))
    (train_x, train_y), (test_x, test_y) = dataset_func(**info.dataset_params)  # type: ignore

    print("Building {} model...".format(info.model_name))
    classifier = model_class(**info.model_params)  # type: ignore

    print("Training...")
    classifier.train(train_x, train_y)

    print("Evaluation...")
    predicted_y = classifier.predict(train_x)

    print("Results:")
    metrics = ClassificationMetrics(train_y, predicted_y)
    print(metrics.get_header())
    print(metrics.to_table_row())
    print(metrics.confusion_matrix)


def verify_choice(name: str, choices: Dict[str, Any], choice: str) -> None:
    if choice not in choices:
        print("{} '{}' not registered.".format(name, choice))
        print("Available choices:")
        print()
        for name in sorted(choices):
            print('- {}'.format(name))
        print()
        sys.exit()

def verify_help(obj: Any, params: Dict[str, Any]) -> None:
    if '--help' in params:
        print(obj.__doc__)
        sys.exit()

class TestMetadata:
    def __init__(self, dataset_name: str, 
                       dataset_params: Dict[str, Primitive],
                       model_name: str, 
                       model_params: Dict[str, Primitive]) -> None:
        self.dataset_name = dataset_name
        self.dataset_params = dataset_params
        self.model_name = model_name
        self.model_params = model_params

    def __str__(self) -> str:
        return '{} -> {}\n{} -> {}'.format(
                self.dataset_name, self.dataset_params,
                self.model_name, self.model_params)

def parse_args(args: List[str]) -> TestMetadata:
    index = 0

    dataset_name = args[index]
    index += 1

    index, dataset_params = parse_arg_list(index, args)

    model_name = args[index]
    index += 1

    index, model_params = parse_arg_list(index, args)

    return TestMetadata(dataset_name, dataset_params, model_name, model_params)

def parse_arg_list(index: int, args: List[str]) -> Tuple[int, Dict[str, Primitive]]:
    out = {}  # type: Dict[str, Any]   # Can't use Primitive here due to coercion bug?
    while index < len(args) and args[index].startswith('--'):
        if args[index] == '--help':
            out['--help'] = True
            index += 1
        else:
            out[args[index][2:]] = parse_arg(args[index + 1])
            index += 2
    return index, out

def parse_arg(blob: str) -> Primitive:
    if blob == 'True':
        return True
    if blob == 'False':
        return False
    try:
        return int(blob)
    except ValueError:
        pass
    try:
        return float(blob)
    except ValueError:
        pass
    return blob


if __name__ == '__main__':
    main()
