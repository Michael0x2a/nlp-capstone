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
    base_log_dir = "runs/filter/run{}"

    # Core methods that must be implemented
    def __init__(self, split_by_word: bool = False,
                       restore_from: Optional[str] = None,
                       run_num: Optional[int] = None) -> None:
        super().__init__(restore_from, run_num)
        self.split_by_word = split_by_word

    def _get_parameters(self) -> Dict[str, Any]:
        return {}

    def _save_model(self, path: str) -> None:
        pass

    def _restore_model(self, path: str) -> None:
        pass

    def train(self, xs: List[str], ys: List[int], **params: Any) -> None:
        pass

    def predict(self, xs: List[str]) -> List[List[float]]:
        out = []
        for sentence in xs:
            sentence = sentence.lower()
            attack = False
            if self.split_by_word:
                split = sentence.split(' ')
                for word in banlist:
                    if word in split:
                        attack = True
                        break
            else:
                for word in banlist:
                    if word in sentence:
                        attack = True
                        break
            if attack:
                out.append([0.0, 1.0])
            else:
                out.append([1.0, 0.0])
        return np.array(out)

