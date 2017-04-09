from typing import Dict, Any, List
import statistics

RevId = int
WorkerId = int
JsonDict = Dict[str, Any]

class Annotation:
    def __init__(self, rev_id: RevId,
                       worker_id: WorkerId,
                       quoting_attack: float,
                       recipient_attack: float,
                       third_party_attack: float,
                       other_attack: float,
                       attack: float) -> None:
        self.rev_id = rev_id
        self.worker_id = worker_id
        self.quoting_attack = quoting_attack
        self.recipient_attack = recipient_attack
        self.third_party_attack = third_party_attack
        self.other_attack = other_attack
        self.attack = attack

    def to_json(self) -> JsonDict:
        return {'ri': self.rev_id, 'wi': self.worker_id, 'qa': self.quoting_attack,
                'ra': self.recipient_attack, 'ta': self.third_party_attack,
                'oa': self.other_attack, 'a': self.attack}

    @staticmethod
    def from_json(raw: Any) -> 'Annotation':
        return Annotation(raw['ri'], raw['wi'], raw['qa'],
                          raw['ra'], raw['ta'], raw['oa'], raw['a'])

class Comment:
    def __init__(self, rev_id: int,
                       comment: str,
                       logged_in: bool,
                       namespace: str,
                       sample: str,
                       annotations: List[Annotation]) -> None:
        self.rev_id = rev_id
        self.comment = comment
        self.logged_in = logged_in
        self.namespace = namespace
        self.sample = sample
        self.annotations = annotations

        self.average = Annotation(
                self.rev_id,
                -1,
                statistics.mean([a.quoting_attack for a in self.annotations]),
                statistics.mean([a.recipient_attack for a in self.annotations]),
                statistics.mean([a.third_party_attack for a in self.annotations]),
                statistics.mean([a.other_attack for a in self.annotations]),
                statistics.mean([a.attack for a in self.annotations]))

    def to_json(self) -> JsonDict:
        return {'ri': self.rev_id, 'c': self.comment, 'li': self.logged_in,
                'ns': self.namespace, 's': self.sample, 
                'a': [ann.to_json() for ann in self.annotations]}

    @staticmethod
    def from_json(raw: Any) -> 'Comment':
        return Comment(raw['ri'], raw['c'], raw['li'],
                       raw['ns'], raw['s'],
                       [Annotation.from_json(ann) for ann in raw['a']])

TrainData = List[Comment]
DevData = List[Comment]
TestData = List[Comment]

