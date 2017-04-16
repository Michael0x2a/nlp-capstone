'''
This python module contains code needed to load data from any of our
datasets (though currently it works only for the Wikipedia detox
data).

Example of usage:

    from parsing import load_raw_data
    train_data, dev_data, test_data = load_raw_data()

Each dataset is a list of custom_types.Comment objects.

Note: this module was written before I realized that the Wikipedia
dataset contained more then just the detox data, so this will likely
need to be refactored later.
'''
from typing import List, Tuple, Any
import os.path
import json
import statistics

import pandas as pd

from custom_types import *

def load_from_json(path: str) -> List[Comment]:
    with open(path, 'r') as stream:
        return [Comment.from_json(c) for c in json.load(stream)]

def save_as_json(path: str, comments: List[Comment]):
    blob = [c.to_json() for c in comments]
    with open(path, 'w') as stream:
        json.dump(blob, stream)

def load_raw_data(folder_path: str = 'data/wikipedia-detox-data-v6') -> Tuple[TrainData, DevData, TestData]:
    train_path = os.path.join(folder_path, 'train.json')
    dev_path = os.path.join(folder_path, 'dev.json')
    test_path = os.path.join(folder_path, 'test.json')
    if os.path.isfile(train_path):
        return load_from_json(train_path), load_from_json(dev_path), load_from_json(test_path)

    comments_path = os.path.join(folder_path, 'attack_annotated_comments.tsv')
    annotations_path = os.path.join(folder_path, 'attack_annotations.tsv')

    comments = pd.read_csv(comments_path, sep='\t', index_col=0)
    annotations = pd.read_csv(annotations_path, sep='\t')

    comments['comment'] = comments['comment'].apply(lambda x: x.replace("NEWLINE_TOKEN", " ").replace("TAB_TOKEN", " "))
    pre_group = annotations.groupby('rev_id')

    def extract(split) -> List[Comment]:
        rows = comments[comments['split'] == split]
        out = []
        for row in rows.itertuples():
            anns = []
            for ann in pre_group.get_group(row.Index).itertuples():
                anns.append(Annotation(
                    int(ann.rev_id),
                    int(ann.worker_id),
                    float(ann.quoting_attack),
                    float(ann.recipient_attack),
                    float(ann.third_party_attack),
                    float(ann.other_attack),
                    float(ann.attack)))
            out.append(Comment(
                int(row.Index),
                row.comment,
                bool(row.logged_in),
                row.ns,
                row.sample,
                anns))
        return out
        
            
    train, dev, test = extract('train'), extract('dev'), extract('test')
    save_as_json(train_path, train)
    save_as_json(dev_path, dev)
    save_as_json(test_path, test)
    return train, dev, test

if __name__ == '__main__':
    print("Starting")
    train, dev, test = load_raw_data()
    print("Done!")
    print(len(train))
    print(len(dev))
    print(len(test))
    attacks = [c.average.attack for c in train]
    print("foo")
    print(statistics.mean(attacks))

