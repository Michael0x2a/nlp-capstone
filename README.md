# nlp-capstone

## Setup

This project is designed to use Python 3.5, specifically (Tensorflow isn't yet
compatible with Python 3.6+).

Installation:

    python3.5 -m pip install -r requirements.txt

Note: when running code, you should be within the `abuse` folder.


## Useful utilities:

### To run the cmd line tool:

Run:

    python3.5 cmd.py [dataset-name] [dataset-params] [model-name] [model-parms]

The params will be forwarded directly as arguments into the dataset
and model names respectively.

The params must be in the form:

    --param_name arg

The types of the args are automatically inferred.

So for example, to run the RNN model using the wikipedia dataset (specifically,
the toxicity dataset), setting the number of epoches to 7 and all other params
the same as the default, you would run:

    python3.5 cmd.py wikipedia --category toxicity rnn --epoch 7


### Regenerating json data caches

To regenerate the cached data json files for a particular data type:

    python3.5 -m data_extraction.[dataset_name].parsing

For example, to regenerate the wikipedia data, run:

    python3.5 -m data_extraction.wikipedia.parsing


### Typechecking

Run:

    mypy [path-to-file.py]

To typecheck the entire project, run:

    mypy ../abuse

...which is a bit of a hack, but whatever.

