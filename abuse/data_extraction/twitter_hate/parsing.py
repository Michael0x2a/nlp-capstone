from data_extraction.twitter_hate.datatypes import TwitterAnnotation
from typing import TypeVar, Tuple, List
import random

import csv
import utils.file_manip as fmanip

Ann = TwitterAnnotation

def load_raw_data(folder_path: str, prefix: str) -> List[TwitterAnnotation]:
    out_list = []
    path = fmanip.join(folder_path, prefix + ".csv")
    with open(path, 'r', encoding='utf-8', errors='ignore') as stream:
        reader = csv.reader(stream, delimiter=',', quotechar='"')
        next(reader)  # discard header
        for row in reader:
            out_list.append(TwitterAnnotation.from_row(row))
    return out_list

def load_all_raw_data(folder_path: str, 
                      prefix: str) -> Tuple[List[Ann], List[Ann], List[Ann]]:
    out = load_raw_data(folder_path, prefix)

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
    a,b , c = load_all_raw_data('data/twitter-hate-speech', 'twitter-hate-speech')
    print(len(a), len(b), len(c))




