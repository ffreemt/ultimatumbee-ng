"""Align via ubee,"""
# pylint: disable=
from typing import Iterable, List, Tuple
from itertools import zip_longest

from logzero import logger
from ubee.uclas import uclas
from icecream import ic


def ubee(
    sents_zh: Iterable,
    sents_en: Iterable,
    thresh: float = 0.5,
) -> Tuple[List[Tuple[str, str, float]], List[Tuple[str, str]]]:
    """Align blocks.

    Args:
        sents_zh: list of text, can be any langauge supported by clas-l-user
        sents_zh: ditto
    Returns:
        three tuples of aligned blocked
        leftovers (unaligned)
    """
    res = []
    labels = [*sents_en]

    lo1 = []
    lo2 = labels[:]

    for seq in sents_zh:
        ic(seq)
        label, likelihood = uclas(seq, labels, thresh=thresh)
        if label:
            likelihood = round(float(likelihood), 2)
            res.append((seq, label, likelihood))
            try:
                lo2.remove(label)
            except Exception as exc:
                logger.error(exc)
                logger.info("seq: %s, lable: %s", seq, label)
        else:
            lo1.append(seq)
    return res, [*zip_longest(lo1, lo2)]
