from typing import List, TypeVar, Generic, Any
import statistics  # type: ignore

from custom_types import *

RevId = int
WorkerId = int

TAnnotation = TypeVar('TAnnotation')

class Comment(Generic[TAnnotation]):
    def __init__(self, rev_id: int,
                       comment: str,
                       logged_in: bool,
                       namespace: str,
                       sample: str,
                       annotations: List[TAnnotation],
                       average: TAnnotation) -> None:
        self.rev_id = rev_id
        self.comment = comment
        self.logged_in = logged_in
        self.namespace = namespace
        self.sample = sample
        self.annotations = annotations
        self.average = average

    def to_json(self) -> JsonDict:
        return {'1': self.rev_id, '2': self.comment, '3': self.logged_in,
                '4': self.namespace, '5': self.sample, 
                '6': [ann.to_json() for ann in self.annotations],  # type: ignore
                '7': self.average.to_json()}  # type: ignore

    @classmethod
    def from_json(cls, raw: JsonDict, ann_class: Any) -> Any:
        return Comment(raw['1'], raw['2'], raw['3'],
                       raw['4'], raw['5'],
                       [ann_class.from_json(ann) for ann in raw['6']],
                       ann_class.from_json(raw['7']))  # type: ignore


class AttackAnnotation:
    def __init__(self, rev_id: RevId,
                       worker_id: WorkerId,
                       quoting_attack: float,
                       recipient_attack: float,
                       third_party_attack: float,
                       other_attack: float,
                       attack: float) -> None:
        '''The attack parameters are either 1.0 or 0.0, unless
        this annotation is being used as an average.'''
        self.rev_id = rev_id
        self.worker_id = worker_id
        self.quoting_attack = quoting_attack
        self.recipient_attack = recipient_attack
        self.third_party_attack = third_party_attack
        self.other_attack = other_attack
        self.attack = attack

    def to_json(self) -> JsonDict:
        return {'1': self.rev_id, '2': self.worker_id, '3': self.quoting_attack,
                '4': self.recipient_attack, '5': self.third_party_attack,
                '6': self.other_attack, '7': self.attack}

    @classmethod
    def from_json(cls, raw: JsonDict) -> 'AttackAnnotation':
        return AttackAnnotation(raw['1'], raw['2'], raw['3'],
                          raw['4'], raw['5'], raw['6'], raw['7'])

    @classmethod
    def parse_row(cls, row: Any) -> 'AttackAnnotation':
        return AttackAnnotation(
                    int(row.rev_id),
                    int(row.worker_id),
                    float(row.quoting_attack),
                    float(row.recipient_attack),
                    float(row.third_party_attack),
                    float(row.other_attack),
                    float(row.attack))

    @classmethod
    def average(cls, rev_id: RevId, annotations: List['AttackAnnotation']) -> 'AttackAnnotation':
        return AttackAnnotation(
                rev_id,
                -1,
                statistics.mean([a.quoting_attack for a in annotations]),
                statistics.mean([a.recipient_attack for a in annotations]),
                statistics.mean([a.third_party_attack for a in annotations]),
                statistics.mean([a.other_attack for a in annotations]),
                statistics.mean([a.attack for a in annotations]))

class AggressionAnnotation:
    def __init__(self, rev_id: RevId,
                       worker_id: WorkerId,
                       aggression_score: float,
                       aggression: float) -> None:
        '''Aggression score is from -2 (very aggressive) to 2 (very friendly);
        aggression is either 0.0 or 1.0, unless this instance is being used
        as an average.'''
        self.rev_id = rev_id
        self.worker_id = worker_id
        self.aggression_score = aggression_score
        self.aggression = aggression

    def to_json(self) -> JsonDict:
        return {'1': self.rev_id, '2': self.worker_id,
                '3': self.aggression_score, '4': self.aggression}

    @classmethod
    def from_json(cls, raw: Any) -> 'AggressionAnnotation':
        return AggressionAnnotation(raw['1'], raw['2'], raw['3'], raw['4'])

    @classmethod
    def parse_row(cls, row: Any) -> 'AggressionAnnotation':
        return AggressionAnnotation(
                    int(row.rev_id),
                    int(row.worker_id),
                    float(row.aggression_score),
                    float(row.aggression))

    @classmethod
    def average(cls, rev_id: RevId, annotations: List['AggressionAnnotation']) -> 'AggressionAnnotation':
        return AggressionAnnotation(
                rev_id,
                -1,
                statistics.mean([a.aggression_score for a in annotations]),
                statistics.mean([a.aggression for a in annotations]))

class ToxicityAnnotation:
    def __init__(self, rev_id: RevId,
                       worker_id: WorkerId,
                       toxicity_score: float,
                       toxicity: float) -> None:
        '''Toxicity score is from -2 (very toxic) to 2 (very healthy);
        toxicity is either 0.0 or 1.0, unless this instance is being used
        as an average.'''
        self.rev_id = rev_id
        self.worker_id = worker_id
        self.toxicity_score = toxicity_score
        self.toxicity = toxicity

    def to_json(self) -> JsonDict:
        return {'ri': self.rev_id, 'wi': self.worker_id,
                'ts': self.toxicity_score, 'tg': self.toxicity}

    @classmethod
    def from_json(cls, raw: JsonDict) -> 'ToxicityAnnotation':
        return ToxicityAnnotation(raw['ri'], raw['wi'], raw['ts'], raw['tg'])

    @classmethod
    def parse_row(cls, row: Any) -> 'ToxicityAnnotation':
        return ToxicityAnnotation(
                    int(row.rev_id),
                    int(row.worker_id),
                    float(row.toxicity_score),
                    float(row.toxicity))

    @classmethod
    def average(cls, rev_id: RevId, annotations: List['ToxicityAnnotation']) -> 'ToxicityAnnotation':
        return ToxicityAnnotation(
                rev_id,
                -1,
                statistics.mean([a.toxicity_score for a in annotations]),
                statistics.mean([a.toxicity for a in annotations]))

