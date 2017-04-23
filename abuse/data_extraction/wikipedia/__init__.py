from typing import Tuple, List
from data_extraction.wikipedia.datatypes import Comment, AttackAnnotation, AggressionAnnotation, ToxicityAnnotation
from data_extraction.wikipedia.parsing import load_raw_data
import utils.file_manip as fmanip

AttackData = List[Comment[AttackAnnotation]]
AggressionData = List[Comment[AggressionAnnotation]]
ToxicityData = List[Comment[ToxicityAnnotation]]

def load_attack_data(prefix: str = 'wikipedia-attack-data-v6',
        small: bool = False) -> Tuple[AttackData, AttackData, AttackData]:
    if small:
        prefix += '-small'
    path = fmanip.join('data', prefix)
    return load_raw_data(path, 'attack', AttackAnnotation)

def load_aggression_data(prefix: str = 'wikipedia-aggression-data-v?',
        small: bool = False) -> Tuple[AggressionData, AggressionData, AggressionData]:
    if small:
        prefix += '-small'
    path = fmanip.join('data', prefix)
    return load_raw_data(path, 'aggression', AggressionAnnotation)

def load_toxicity_data(prefix: str = 'wikipedia-toxicity-data-v2',
        small: bool = False) -> Tuple[ToxicityData, ToxicityData, ToxicityData]:
    if small:
        prefix += '-small'
    path = fmanip.join('data', prefix)
    return load_raw_data(path, 'toxicity', ToxicityAnnotation)

__all__ = [
        'Comment', 'AttackAnnotation', 'AggressionAnnotation', 'ToxicityAnnotation',
        'load_attack_data', 'load_aggression_data', 'load_toxicity_data',
        'AttackData', 'AggressionData', 'ToxicityData',
]
