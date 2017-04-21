'''
Contains a variety of utility functions used for manipulating files.
'''

import os
import os.path
import json

from custom_types import JsonDict

def join(*path_fragments: str) -> str:
    normalized = [os.path.normpath(frag) for frag in path_fragments]
    return os.path.join(*normalized)

def write_nice_json(chunk: JsonDict, path: str) -> None:
    with open(path, 'w') as stream:
        json.dump(chunk, path, indent=4, sort_key=True)

def load_json(path: str) -> JsonDict:
    with open(path, 'r') as stream:
        return json.load(stream)

def ensure_folder_exists(path: str) -> None:
    os.makedirs(path, exists_ok=True)
