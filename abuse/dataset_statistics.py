from custom_types import *
from parsing import load_raw_data
import nltk
import statistics

def analyze_aspect(title: str, dataset: List[Comment]) -> None:
    attack = []
    ok = []
    para_lengths = []
    for c in dataset:
        if c.average.attack >= 0.5:
            attack.append(c)
        else:
            ok.append(c)
        para_lengths.append(len(nltk.word_tokenize(c.comment)))

    len_attack = len(attack)
    len_ok = len(ok)

    print(title)
    print("    Total size: {}".format(len(dataset)))
    print("    Num ok:     {} ({:.5f}%)".format(len_ok, len_ok / len(dataset) * 100))
    print("    Num attack: {} ({:.5f}%)".format(len_attack, len_attack / len(dataset) * 100))
    print("    Avg para len:  {:.5f}".format(statistics.mean(para_lengths)))
    print("    Stdev p len:   {:.5f}".format(statistics.stdev(para_lengths)))
    print("    Para min:      {}".format(min(para_lengths)))
    print("    Para max:      {}".format(max(para_lengths)))
    print()



def analyze(title: str, train: List[Comment], dev: List[Comment], test: List[Comment]) -> None:
    print(title)
    analyze_aspect('train', train)
    analyze_aspect('dev', dev)
    analyze_aspect('test', test)
    print()


def main() -> None:
    train, dev, test = load_raw_data('data/wikipedia-detox-data-v6')
    analyze("FULL", train, dev, test)
    print()

    train, dev, test = load_raw_data('data/wikipedia-detox-data-v6-small')
    analyze("SMALL", train, dev, test)

if __name__ == '__main__':
    main()

