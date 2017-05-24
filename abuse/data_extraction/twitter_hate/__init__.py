from typing import Tuple, List

from data_extraction.twitter_hate.parsing import load_all_raw_data
from data_extraction.twitter_hate.datatypes import TwitterAnnotation

AnnData = List[TwitterAnnotation]

def load_twitter_data() -> Tuple[AnnData, AnnData, AnnData]:
    return load_all_raw_data('data/twitter-hate-speech', 'twitter-hate-speech')

