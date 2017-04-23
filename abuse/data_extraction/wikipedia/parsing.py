'''
This module contains code needed to load data from the Wikipedia
"personal attacks" dataset.
'''
if False:
    # Hack: Python 3.5.1 doesn't have 'Type' yet, so we're
    # working around it.
    from typing import List, Tuple, Any, TypeVar, Type

import os.path
import json
import statistics  # type: ignore

import pandas as pd  # type: ignore

from custom_types import *
from data_extraction.wikipedia.datatypes import *
import utils.file_manip as fmanip

TAnn = TypeVar('TAnn', AttackAnnotation, AggressionAnnotation, ToxicityAnnotation)

def load_from_json(path, ann_cls):
    # type: (str, Type[TAnn]) -> List[Comment[TAnn]]
    with open(path, 'r') as stream:
        return [Comment.from_json(c, ann_cls) for c in json.load(stream)]

def save_as_json(path: str, comments: List[Comment]) -> None:
    blob = [c.to_json() for c in comments]
    with open(path, 'w') as stream:
        json.dump(blob, stream)


def load_raw_data(folder_path, prefix, ann_cls):
    # type: (str, str, Type[TAnn]) -> Tuple[List[Comment[TAnn]], List[Comment[TAnn]], List[Comment[TAnn]]]
    train_path = fmanip.join(folder_path, 'train.json')
    dev_path = fmanip.join(folder_path, 'dev.json')
    test_path = fmanip.join(folder_path, 'test.json')
    if os.path.isfile(train_path):
        return (load_from_json(train_path, ann_cls), 
                load_from_json(dev_path, ann_cls), 
                load_from_json(test_path, ann_cls))

    comments_path = os.path.join(folder_path, prefix + '_annotated_comments.tsv')
    annotations_path = os.path.join(folder_path, prefix + '_annotations.tsv')

    comments = pd.read_csv(comments_path, sep='\t', index_col=0)
    annotations = pd.read_csv(annotations_path, sep='\t')

    comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " ").replace("TAB_TOKEN", " "))
    pre_group = annotations.groupby('rev_id')

    def extract(split: str) -> List[Comment]:
        rows = comments[comments['split'] == split]
        out = []
        for row in rows.itertuples():
            anns = []
            for ann in pre_group.get_group(row.Index).itertuples():
                anns.append(ann_cls.parse_row(ann))
            rev_id = int(row.Index)
            out.append(Comment(
                rev_id,
                row.comment,
                bool(row.logged_in),
                row.ns,
                row.sample,
                anns,
                ann_cls.average(rev_id, anns)))
        return out
        
    train, dev, test = extract('train'), extract('dev'), extract('test')
    save_as_json(train_path, train)
    save_as_json(dev_path, dev)
    save_as_json(test_path, test)

    return train, dev, test

def save_small(prefix, ann_cls, src, dest, percentage):
    # type: (str, Type[TAnn], str, str, float) -> None
    train, dev, test = load_raw_data(src, prefix, ann_cls)

    fmanip.ensure_folder_exists(dest)
    train_path = fmanip.join(dest, 'train.json')
    dev_path = fmanip.join(dest, 'dev.json')
    test_path = fmanip.join(dest, 'test.json')

    if not os.path.isfile(train_path):
        save_as_json(train_path, train[:int(len(train) * percentage)])
        save_as_json(dev_path, dev[:int(len(dev) * percentage)])
        save_as_json(test_path, test[:int(len(test) * percentage)])

if __name__ == '__main__':
    save_small('attack', AttackAnnotation, 'data/wikipedia-attack-data-v6', 'data/wikipedia-attack-data-v6-small', 0.1)
    save_small('toxicity', ToxicityAnnotation, 'data/wikipedia-toxicity-data-v2', 'data/wikipedia-toxicity-data-v2-small', 0.1)

