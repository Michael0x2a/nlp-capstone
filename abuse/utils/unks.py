from typing import List, Dict, Tuple, Optional
import random
import re
from collections import Counter

import nltk  # type: ignore
import nltk.corpus  # type: ignore

# Type aliases, for readability
Paragraph = List[str]
WordId = int
ParagraphVec = List[WordId]
Label = int
VocabMap = Dict[str, WordId]

stop_words = set(nltk.corpus.stopwords.words('english'))

# Copied from https://mathiasbynens.be/demo/url-regex
# using @imme_emosol's implementation
_url = re.compile('(https?|ftp)://(-\\.)?([^\\s/?\\.#-]+\\.?)+(/[^\\s]*)?')
_missing_space_after_period = re.compile('([a-z])\\.([a-zA-Z])')
_quotes_at_start = re.compile("''([a-zA-Z])")
_colons_at_start = re.compile('\\:+([a-zA-Z0-9])')

def to_words(inputs: str) -> List[str]:
    #inputs = re.sub(_url, '$UNK-URL', inputs)
    inputs = inputs.replace('=====', '')
    inputs = inputs.replace('====', '')
    inputs = inputs.replace('===', '')
    inputs = inputs.replace('==', '')
    inputs = inputs.replace('`', "'")
    inputs = re.sub(_quotes_at_start, (lambda match: "'' {}".format(match.group(1))), inputs)
    inputs = re.sub(_colons_at_start, (lambda match: match.group(1)), inputs)
    words = nltk.word_tokenize(inputs)
    out = []
    for word in words:
        if word.startswith('//') and word != '//':
            out.append('$UNK-URL')
        else:
            out.append(word)
    
    return out

def truncate_and_pad(paragraph: Paragraph, max_length: int) -> Paragraph:
    # Subtract 2 so we have space for the start and end tokens
    length = min(len(paragraph), max_length - 2)
    if length < max_length - 2:
        padding = max_length - 2 - length
    else:
        padding = 0

    out = ["$START"] + paragraph[:length] + ["$END"] + (["$PADDING"] * padding)
    return out

def make_vocab_mapping(x: List[Paragraph],
                       max_vocab_size: Optional[int] = None) -> Dict[str, WordId]:
    freqs = Counter()  # type: Counter[str]
    words_set = set()
    for paragraph in x:
        for word in paragraph:
            freqs[word] = freqs.get(word, 0) + 1
            words_set.add(word)
    out = {'$UNK': 0}
    count = 1
    if max_vocab_size is None:
        max_vocab_size = len(freqs)
    print("Actual vocab size: {}".format(len(freqs)))
    for key, num in freqs.most_common(max_vocab_size - 1):
        #if num == 1:
        #    continue
        out[key] = count
        count += 1
    with open("unks.txt", "w") as stream:
        for word in words_set:
            if word not in out:
                stream.write(word)
                stream.write("\n")
    return out

def vectorize_paragraph(vocab_map: Dict[str, WordId], para: Paragraph) -> List[WordId]:
    unk_id = vocab_map['$UNK']
    return [vocab_map.get(word, unk_id) for word in para]

def prep_train(xs: List[str], comment_size: int, vocab_size: int) -> Tuple[List[List[int]], List[int], VocabMap]:
    x_data_raw = [truncate_and_pad(to_words(x), comment_size) for x in xs]
    x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
    vocab_map = make_vocab_mapping(x_data_raw, vocab_size)
    x_final = [vectorize_paragraph(vocab_map, para) for para in x_data_raw]

    return x_final, x_lengths, vocab_map

def prep_train_char(xs: List[str], comment_size: int, vocab_size: int) -> Tuple[List[List[int]], List[int], VocabMap]:
    x_data_raw = [truncate_and_pad(list(x), comment_size) for x in xs]
    x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
    vocab_map = make_vocab_mapping(x_data_raw, vocab_size)
    x_final = [vectorize_paragraph(vocab_map, para) for para in x_data_raw]

    return x_final, x_lengths, vocab_map

def shuffle(xs: List[List[int]], x_lengths: List[int], ys: List[int]) -> Tuple[List[List[int]], List[int], List[int]]:
    indices = list(range(len(xs)))
    random.shuffle(indices)
    x_final_new = [xs[i] for i in indices]
    x_lengths_new = [x_lengths[i] for i in indices]
    ys_new = [ys[i] for i in indices]

    return x_final_new, x_lengths_new, ys_new

def prep_test(xs: List[str], comment_size: int, vocab_map: VocabMap) -> Tuple[List[List[int]], List[int]]:
    x_data_raw = [truncate_and_pad(to_words(x), comment_size) for x in xs]
    x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
    x_final = [vectorize_paragraph(vocab_map, para) for para in x_data_raw]
    return x_final, x_lengths

def prep_test_char(xs: List[str], comment_size: int, vocab_map: VocabMap) -> Tuple[List[List[int]], List[int]]:
    x_data_raw = [truncate_and_pad(list(x), comment_size) for x in xs]
    x_lengths = [x.index('$PADDING') if x[-1] == '$PADDING' else len(x) for x in x_data_raw]
    x_final = [vectorize_paragraph(vocab_map, para) for para in x_data_raw]
    return x_final, x_lengths

