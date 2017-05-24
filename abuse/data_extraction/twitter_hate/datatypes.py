from typing import List
LabelType = int

HATE = 0
OFFENSIVE = 1
NEITHER = 2

class TwitterAnnotation:
    def __init__(self, tweet_id: str,
                       num_annotators: int,
                       num_decide_hate: int,
                       num_decide_offensive: int,
                       num_decide_neither: int,
                       label: LabelType,
                       text: str) -> None:
        self.tweet_id = tweet_id
        self.num_annotators = num_annotators
        self.num_decide_hate = num_decide_hate
        self.num_decide_offensive = num_decide_offensive
        self.num_decide_neither = num_decide_neither
        self.label = label
        self.text = text
        self.is_bad = 0 if label == NEITHER else 1

    @classmethod
    def from_row(cls, row: List[str]) -> None:
        return TwitterAnnotation(
                row[0],
                int(row[1]),
                int(row[2]),
                int(row[3]),
                int(row[4]),
                int(row[5]),
                row[6])

