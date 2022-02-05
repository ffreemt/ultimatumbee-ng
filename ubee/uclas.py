"""Define uclas."""
# pylint: disable=invalid-name

from typing import List, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from joblib import Memory

from model_pool import fetch_check_aux  # pylint: disable=import-error
from model_pool.model_s import load_model_s  # pylint: disable=import-error
from model_pool.load_model import load_model  # pylint: disable=import-error

import logzero
from logzero import logger

logzero.loglevel(20)

fetch_check_aux("/home/user")
model_s = load_model_s()
clas = load_model("clas-l-user")

location = "./cachedir"
memory = Memory(location, verbose=0)


@memory.cache
def cached_clas(*args, **kw):
    """Cache clas-l-user."""
    return clas(*args, **kw)


# cached_clas = memory.cache(cached_clas)


@memory.cache
def encode(*args, **kw):
    """Cache model_s.encode."""
    return model_s.encode(*args, **kw)


def uclas(
    seq: str,
    labels: Union[List[str], np.ndarray, Tuple[str, ...]],
    thresh: float = 0.5,
    multi_label: bool = False,
) -> Tuple[str, Union[float, str]]:
    """Classify seq with a filter.

    if clas > thresh, return
    if clas * csim > thresh return
    if csim > thresh return
    return ""
    """
    # _ = clas(seq, labels, multi_label=multi_label)
    _ = cached_clas(seq, labels, multi_label=multi_label)

    logger.debug("1 %s, %s", _.get("labels")[0], round(_.get("scores")[0], 2))

    if _.get("scores")[0] > thresh:
        return _.get("labels")[0], round(_.get("scores")[0], 2)

    _ = dict(zip(_.get("labels"), _.get("scores")))

    corr = np.array([_.get(elm) for elm in labels])

    csim = cosine_similarity(encode([seq]), encode(labels))

    corr = corr * csim

    logger.debug("2 %s, %s", corr.argmax(), round(corr.max(), 2))

    if corr.max() > thresh:
        return labels[corr.argmax()], round(corr.max(), 2)

    logger.debug("3 %s, %s, %s", csim.argmax(), round(csim.max(), 2), thresh / 2)

    logger.debug("T or F: %s", csim.max() > (thresh / 2))
    if csim.max() > (thresh / 2):
        return labels[csim.argmax()], round(csim.max(), 2)

    return "", ""
