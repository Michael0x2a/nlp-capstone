'''
Usage:

    python3.5 -m webapp

'''
import time

import flask
from flask_cors import CORS, cross_origin

from models.model import Model
from models.bag_of_words import BagOfWordsClassifier
#from models.rnn_classifier import RnnClassifier

# Load classifiers into global scope (may take a while)
clf1 = BagOfWordsClassifier.restore_from_saved('attack_bag_of_words')
#clf2 = RnnClassifier.restore_from_saved('attack_rnn_small_epoch6')


# Wire together webapp
app = flask.Flask(__name__)
CORS(app)

@app.route(r'/')
def index():
    return flask.render_template('index.html')

@app.route(r'/api/classify', methods=['POST'])
def api_classify():
    # force=True: Parse the input as JSON even if the correct
    # Content-Type isn't set.
    data = flask.request.get_json(force=True)
    text = data['comment']

    print(data)

    out = {
        'attack': {
            'bag_of_words': fetch(clf1, text),
            'lr': placeholder(text),
            'rnn': placeholder(text),
        },
        'aggression': {
            'bag_of_words': placeholder(text),
            'lr': placeholder(text),
            'rnn': placeholder(text),
        },
        'toxicity': {
            'bag_of_words': placeholder(text),
            'lr': placeholder(text),
            'rnn': placeholder(text),
        },
    }

    return flask.jsonify(out)

def placeholder(text: str) -> None:
    return {"prob_ok": 0.5, "prob_attack": 0.5, "time": 0.0}

def fetch(clf: Model, text: str) -> None:
    start = time.time()
    out = clf.predict_single(text)
    end = time.time() - start
    return {
            "prob_ok": out[0],
            "prob_attack": out[1],
            "time": end,
    }


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

