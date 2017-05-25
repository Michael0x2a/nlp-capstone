'''
Usage:

    python3.5 -m webapp

'''
import time

import flask
from flask_cors import CORS, cross_origin

import tensorflow as tf

from models.model import Model
from models.bag_of_words import BagOfWordsClassifier
from models.profanity_filter import ProfanityFilterClassifier
from models.rnn_classifier import RnnClassifier
from models.logistic_copy import CopiedClassifier 
from models.rnn_filter_aggression import build_rewriter

# Load classifiers into global scope (may take a while)
profanity_clf = ProfanityFilterClassifier(split_by_word=True)

bag_clf_attack = BagOfWordsClassifier.restore_from_saved(path='core_models_2/attack_bag_of_words')
bag_clf_toxicity = BagOfWordsClassifier.restore_from_saved(path='core_models_2/toxicity_bag_of_words')
bag_clf_aggression = BagOfWordsClassifier.restore_from_saved(path='core_models_2/aggression_bag_of_words')
bag_clf_stanford = BagOfWordsClassifier.restore_from_saved(path='core_models_2/stanford_bag_of_words')

lr_clf_attack = CopiedClassifier.restore_from_saved(path='core_models_2/attack_lr')
lr_clf_toxicity = CopiedClassifier.restore_from_saved(path='core_models_2/toxicity_lr')
lr_clf_aggression = CopiedClassifier.restore_from_saved(path='core_models_2/aggression_lr')
lr_clf_stanford = CopiedClassifier.restore_from_saved(path='core_models_2/stanford_lr')


with tf.Graph().as_default() as g:
    rnn_clf_attack = RnnClassifier.restore_from_saved(path='core_models_2/attack_rnn')
with tf.Graph().as_default() as g:
    rnn_clf_toxicity = RnnClassifier.restore_from_saved(path='core_models_2/toxicity_rnn')
with tf.Graph().as_default() as g:
    rnn_clf_aggression = RnnClassifier.restore_from_saved(path='core_models_2/aggression_rnn')
with tf.Graph().as_default() as g:
    rnn_clf_stanford = RnnClassifier.restore_from_saved(path='core_models_2/stanford_rnn')

rewriter = build_rewriter()


'''
with tf.Graph().as_default() as g:
    lr_clf_attack = LogisticClassifier.restore_from_saved('core_models/attack_lr')
with tf.Graph().as_default() as g:
    lr_clf_toxicity = LogisticClassifier.restore_from_saved('core_models/toxicity_lr')
with tf.Graph().as_default() as g:
    lr_clf_aggression = LogisticClassifier.restore_from_saved('core_models/aggression_lr')'''


# Wire together webapp
app = flask.Flask(__name__)
CORS(app)

@app.route(r'/')
def index():
    return flask.render_template('index.html')

@app.route(r'/api/rewrite', methods=['POST'])
def api_rewrite():
    # force=True: Parse the input as JSON even if the correct
    # Content-Type isn't set.
    data = flask.request.get_json(force=True)
    text = data['comment']

    print(data)

    out = rewriter(text)
    return flask.jsonify({'comment': out})



@app.route(r'/api/classify', methods=['POST'])
def api_classify():
    # force=True: Parse the input as JSON even if the correct
    # Content-Type isn't set.
    data = flask.request.get_json(force=True)
    text = data['comment']

    print(data)

    out = {
        'attack': {
            'profanity': fetch(profanity_clf, text),
            'bag_of_words': fetch(bag_clf_attack, text),
            'lr': fetch(lr_clf_attack, text),
            'rnn': fetch(rnn_clf_attack, text),
        },
        'aggression': {
            'profanity': fetch(profanity_clf, text),
            'bag_of_words': fetch(bag_clf_aggression, text),
            'lr': fetch(lr_clf_aggression, text),
            'rnn': fetch(rnn_clf_aggression, text),
        },
        'toxicity': {
            'profanity': fetch(profanity_clf, text),
            'bag_of_words': fetch(bag_clf_toxicity, text),
            'lr': fetch(lr_clf_toxicity, text),
            'rnn': fetch(rnn_clf_toxicity, text),
        },
        'stanford': {
            'profanity': fetch_stanford(profanity_clf, text),
            'bag_of_words': fetch_stanford(bag_clf_stanford, text),
            'lr': fetch_stanford(lr_clf_stanford, text),
            'rnn': fetch_stanford(rnn_clf_stanford, text),
        },
    }

    print(out)

    return flask.jsonify(out)

def placeholder(text: str) -> None:
    return {"prob_ok": 0.5, "prob_attack": 0.5, "time": 0.0}

def fetch(clf: Model, text: str) -> None:
    start = time.time()
    out = clf.predict_single(text)
    end = time.time() - start
    return {
            "prob_ok": float(out[0]),
            "prob_attack": float(out[1]),
            "time": end,
    }

def fetch_stanford(clf: Model, text: str) -> None:
    start = time.time()
    out = clf.predict_single(text)
    end = time.time() - start
    return {
            "prob_ok": float(out[1]),
            "prob_attack": float(out[0]),
            "time": end,
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)

