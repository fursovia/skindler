from itertools import islice
from typing import Iterable, Iterator, List, TypeVar, Dict, Any
from skindler.modules.metrics import ALL_METRICS


A = TypeVar("A")


def lazy_groups_of(iterable: Iterable[A], group_size: int) -> Iterator[List[A]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    iterator = iter(iterable)
    while True:
        s = list(islice(iterator, group_size))
        if len(s) > 0:
            yield s
        else:
            break


def count_metrics(one_list: List[str], second_list: List[str]) -> Dict[str, Any]:
    result = {}
    for metric in ['bleu', 'meteor', 'chrf', 'bertscore',
                   'calculate_wer_corpus', 'calculate_paraphrase_similarity']:
        result[metric] = ALL_METRICS[metric](one_list, second_list)
    result['wer'] = result.pop('calculate_wer_corpus')
    result['par.similarity'] = result.pop('calculate_paraphrase_similarity')
    return result