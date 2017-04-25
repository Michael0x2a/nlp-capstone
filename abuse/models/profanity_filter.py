from typing import Optional, Dict, Tuple, cast, Any

from custom_types import *
from models.model import Model 
import utils.file_manip as fmanip
import numpy as np

# From http://www.bannedwordlist.com/lists/swearWords.txt
banlist = [
    'anal', 'anus', 'arse', 'ass', 'ballsack', 'balls', 'bastard', 'bitch', 'biatch', 
    'bloody', 'blowjob', 'blow job', 'bollock', 'bollok', 'boner', 'boob', 'bugger', 
    'bum', 'butt', 'buttplug', 'clitoris', 'cock', 'coon', 'crap', 'cunt', 'damn', 
    'dick', 'dildo', 'dyke', 'fag', 'feck', 'fellate', 'fellatio', 'felching', 'fuck', 
    'f u c k', 'fudgepacker', 'fudge packer', 'flange', 'Goddamn', 'God damn', 'hell', 
    'homo', 'jerk', 'jizz', 'knobend', 'knob end', 'labia', 'lmao', 'lmfao', 'muff', 
    'nigger', 'nigga', 'penis', 'piss', 'poop', 'prick', 'pube', 'pussy', 
    'queer', 'scrotum', 'sex', 'shit', 's hit', 'sh1t', 'slut', 'smegma', 'spunk', 
    'tit', 'tosser', 'turd', 'twat', 'vagina', 'wank', 'whore', 
]

class ProfanityFilterClassifier(Model[str]):
    # Core methods that must be implemented
    def __init__(self, restore_from: Optional[str] = None) -> None:
        pass

    def get_parameters(self) -> Dict[str, Any]:
        return {}

    def _save_model(self, path: str) -> None:
        pass

    def train(self, xs: List[str], ys: List[int], **params: Any) -> None:
        pass

    def predict(self, xs: List[str]) -> List[List[float]]:
        out = []
        for sentence in xs:
            sentence = sentence.lower()
            for word in banlist:
                if word in sentence:
                    out.append([0.0, 1.0])
                    break
            else:
                out.append([1.0, 0.0])
        return np.array(out)

