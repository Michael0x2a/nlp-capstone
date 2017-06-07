'''Command line interface for extracting data. Note: I'm
not using any command line libraries and am just manually
munging sys.argv. This is more out of laziness more then
anything.'''

from typing import Tuple, Dict, Any, Union, List, TypeVar, cast
if False:  # Hack to support Python 3.5.1
    from typing import Type
import sys

import numpy as np  # type: ignore

from models.rnn_language_model import RnnLanguageModel
from models.model import Model
import utils.file_manip as fmanip
from utils.unks import to_words

Primitive = Union[int, float, str, bool]
Data = Tuple[List[str], List[float]]



def load_hedges() -> List[List[str]]:
    with open("hedges.txt", "r") as f:
        hedges = f.readlines()
    return [to_words(x) for x in hedges]

def score(comment: List[List[str]], model: RnnLanguageModel) -> float:
    attack_prob, log_perp =\
            model._run_batch_tokenized([model.output_prob, model.log_perplexity], [comment])
    perp = np.exp(log_perp)
    return perp * attack_prob[0,1] # tune this

def rewrite_k(comment: str, model: RnnLanguageModel, hedges: List[List[str]], k: int=5) -> Tuple[List[List[str]], List[float]]:
    words = to_words(comment)
    edits = []
    scores = []
    for hedge in hedges:
        for i in range(len(words) + 1):
            edit = words[:i] + hedge + words[i:]
            # print("  > ", edit)
            edits.append(edit)
            scores.append(score(edit, model))
    edits = np.array(edits)
    scores = np.array(scores)
    topk = np.argpartition(scores, k)[:k]  # get k lowest scores
    # print("topk", topk)
    topk_sorted = topk[np.argsort(scores[topk])]  # sort by score, lowest first
    # print("topk_sorted", topk_sorted)
    return edits[topk_sorted], scores[topk_sorted]

def main() -> None:
    hedges = load_hedges()

    print("Restoring model...")
    model = RnnLanguageModel.restore_from_saved(run_num=-1)

    while(True):
        print("Enter a comment:")
        comment = input()

        if comment.startswith("SCORE "):
            print("Score:", score(to_words(comment[6:]), model))
            print()
            continue
        if comment.startswith("PERP "):
            print("Perplexity:", model.perplexity([comment[5:]]))
            print()
            continue
        if comment.startswith("ATT "):
            print("Attack prob:", model.predict([comment[4:]]))
            print()
            continue
        if comment.startswith("VOC "):
            print("Vocab:", model.vocab_map[comment[4:]] if comment[4:] in model.vocab_map else "OOV")
            print()
            continue

        print("Rewrites and scores:")
        edits, scores = rewrite_k(comment, model, hedges)
        for e, s in zip(edits, scores):
            print(s, " ".join(e))
        print()

if __name__ == '__main__':
    main()
