from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

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

    test_comment = [c.comment for c in test]
    test_attack = [c.average.attack > 0.5 for c in test]

    clf.fit(train_comment, train_attack)

    auc = roc_auc_score(test_attack, clf.predict_proba(test_comment)[:, 1])

    print('Test ROC AUC: {}'.format(auc))


if __name__ == '__main__':
    train, dev, test = load_raw_data()
    bag_of_words_model(train, dev, test)

