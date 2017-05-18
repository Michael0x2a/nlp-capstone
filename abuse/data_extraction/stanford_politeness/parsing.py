from typing import TypeVar, Tuple, List
if False:
    from typing import Type
import random

import csv
import utils.file_manip as fmanip
from data_extraction.stanford_politeness.datatypes import Annotation, AnnotatedRequest


TAnn = TypeVar('TAnn', bound=Annotation)

def load_raw_data(folder_path, prefix, ann_cls):
    # type: (str, str, Type[TAnn]) -> List[TAnn]
    out_list = []
    with open(fmanip.join(folder_path, prefix + ".csv"), 'r') as stream:
        reader = csv.reader(stream, delimiter=',', quotechar='"')
        next(reader)  # skip header
        for row in reader:
            out_list.append(ann_cls.from_row(row))
    return out_list

def load_all_raw_data(folder_path, prefixes, ann_cls):
    # type: (str, List[str], Type[TAnn]) -> Tuple[List[TAnn], List[TAnn], List[TAnn]]
    out = []
    for prefix in prefixes:
        out.extend(load_raw_data(folder_path, prefix, ann_cls))

    # Make normalized scores fit in range 0 to 1
    scores = [t.normalized_score for t in out]
    old_min = min(scores)
    old_max = max(scores)
    old_range = old_max - old_min
    new_range = 1.0
    for t in out:
        new_val = (t.normalized_score - old_min) * new_range / old_range
        t.normalized_score = new_val

    rand = random.Random()
    rand.seed(0)
    rand.shuffle(out)
    
    length = len(out)
    a = int(length * 0.6)
    b = int(length * 0.8)
    c = length

    train = out[0:a]
    dev = out[a:b]
    test = out[b:c]
    
    return (train, dev, test)

if __name__ == '__main__':
    train, dev, test = load_all_raw_data(
            'data/stanford-politeness',
            ['stack-exchange.annotated', 'wikipedia.annotated'],
            AnnotatedRequest)
    combo = train + dev + test

    print("Lengths")
    print(len(train))
    print(len(dev))
    print(len(test))
    print(len(combo))
    print()

    scores = [t.normalized_score for t in combo]
    print("Scores")
    print(min(scores), max(scores), sum(scores) / len(scores))
    print()

    import nltk
    import statistics
    tlen = [len(nltk.word_tokenize(t.text)) for t in combo]
    print("Text info")
    print(min(tlen), max(tlen), statistics.mean(tlen), statistics.stdev(tlen))
    print()

    clen = [len(t.text) for t in combo]
    print("Text info")
    print(min(clen), max(clen), statistics.mean(clen), statistics.stdev(clen))
    print()

