from typing import Callable, Dict, Any

import glob
import json
import shutil
import os
import os.path
from os.path import join
import re

JsonDict = Dict[str, Any]

def apply_to_para(para: JsonDict, func: Callable[[str], str]) -> JsonDict:
    return {
            'original': func(para['original']),
            'translated': func(para['translated']),
            'section_name': para['section_name'],
    }

def remove_junk(blob: JsonDict) -> JsonDict:
    new_para = []
    for para in blob['paragraphs']:
        o_flag = para['original'] == 'ORIGINAL TEXT'
        t_flag = para['translated'] == 'MODERN TEXT'

        if o_flag or t_flag:
            continue

        new_para.append(apply_to_para(
            para, 
            lambda x: x.replace('$NEWLINE$', '\n')
                       .replace('\u2019', "'")
                       .replace('\u2014', " - ")
                       .replace('\u201c', '"')
                       .replace('\u201d', '"')
                       .replace('\u2018', "'")))

    return {'metadata': blob['metadata'], 'paragraphs': new_para}

def remove_speakers(blob: JsonDict) -> JsonDict:
    regex = re.compile('^([A-Z ]+)$', re.MULTILINE)
    new_para = []
    for para in blob['paragraphs']:
        new_para.append(apply_to_para(
            para,
            lambda x: re.sub(regex, '', x).strip()))

    return {'metadata': blob['metadata'], 'paragraphs': new_para}

def apply(func: Callable[[JsonDict], JsonDict],
          src_folder: str, 
          dest_folder: str) -> None:
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.mkdir(dest_folder)

    for src in glob.glob(join(src_folder, "*.json")):
        dest = src.replace(src_folder, dest_folder)
        with open(src, 'r') as stream:
            blob = json.load(stream)
        out = func(blob)
        with open(dest, 'w') as stream:
            json.dump(out, stream, sort_keys=True, indent=4)

def main():
    apply(remove_junk, join("data", "raw"), join("data", "cleaned"))
    apply(remove_speakers, join("data", "cleaned"), join("data", "no-speakers"))

if __name__ == '__main__':
    main()

