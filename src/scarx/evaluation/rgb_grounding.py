"""Evaluation-only negative image pairing; labels never enter model features."""
import numpy as np


def different_class_pairing(labels):
    y=np.asarray(labels)
    if y.ndim!=1 or not len(y):raise ValueError('Need labels for evaluation pairing')
    classes,counts=np.unique(y,return_counts=True)
    if len(classes)<2 or not np.all(counts==counts[0]):raise ValueError('Balanced multi-class evaluation required')
    order=np.argsort(y,kind='stable');permutation=np.empty(len(y),np.int64)
    permutation[order]=np.roll(order,int(counts[0]))
    if not np.all(y[permutation]!=y):raise ValueError('Pairing did not change every class')
    return permutation
