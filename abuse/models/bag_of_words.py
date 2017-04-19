from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
import sklearn.metrics

from custom_types import *
from parsing import load_raw_data

def bag_of_words_model(train: TrainData, dev: DevData, test: TestData) -> None:
    clf = Pipeline([
        ('vect', CountVectorizer(max_features=10000, ngram_range=(1, 2))),
        ('tfidf', TfidfTransformer(norm='l2')),
        ('clf', LogisticRegression())
    ])

    train_comment = [c.comment for c in train]
    train_attack = [c.average.attack > 0.5 for c in train]

    dev_comment = [c.comment for c in dev]
    dev_attack = [c.average.attack > 0.5 for c in dev]

    clf.fit(train_comment, train_attack)

    # Evaluation
    dev_attack_predicted = clf.predict(dev_comment)

    print("    Accuracy:   {:.6f}".format(sklearn.metrics.accuracy_score(dev_attack, dev_attack_predicted)))
    print("    Precision:  {:.6f}".format(sklearn.metrics.precision_score(dev_attack, dev_attack_predicted)))
    print("    Recall:     {:.6f}".format(sklearn.metrics.recall_score(dev_attack, dev_attack_predicted)))
    print("    F1 score:   {:.6f}".format(sklearn.metrics.f1_score(dev_attack, dev_attack_predicted)))
    print("    AUC score:  {}".format(sklearn.metrics.roc_auc_score(dev_attack, dev_attack_predicted)))
    print("    Confusion matrix:")
    print(sklearn.metrics.confusion_matrix(dev_attack, dev_attack_predicted))
    print()

    count = 0
    for idx, (actual, predicted) in enumerate(zip(dev_attack, dev_attack_predicted)):
        if actual != predicted:
            print(dev[idx].comment)
            print('Attack score={}, is_attack={}, predicted={}'.format(dev[idx].average.attack, actual, predicted))
            print()
            count += 1

        if count == 30:
            break


if __name__ == '__main__':
    data_src_path = 'data/wikipedia-detox-data-v6'
    print("Loading...")
    train, dev, test = load_raw_data(data_src_path)
    print("Training...")
    bag_of_words_model(train, dev, test)

