from typing import Tuple, List

from data_extraction.stanford_politeness.parsing import load_all_raw_data
from data_extraction.stanford_politeness.datatypes import AnnotatedRequest

AnnData = List[AnnotatedRequest]

def load_stanford_data() -> Tuple[AnnData, AnnData, AnnData]:
    return load_all_raw_data(
            'data/stanford-politeness',
            ['stack-exchange.annotated', 'wikipedia.annotated'],
            AnnotatedRequest)

