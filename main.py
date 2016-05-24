#!/usr/bin/python3

from collections import defaultdict
from math import log

samples0 = [
    ((1, 2, 1), 4),
    ((1, 1, 1), 1),
    ((1, 1, 1), 1),
    ((2, 1, 1), 2),
    ((2, 1, 2), 3),
    ((1, 2, 2), 5),
    ((2, 1, 1), 2),
    ((1, 1, 1), 1),
    ((2, 1, 1), 2),
    ((1, 2, 1), 4),
    ((2, 2, 1), 5),
    ((2, 2, 1), 5),
    ((1, 1, 2), 2),
    ((1, 2, 1), 4),
    ((2, 2, 2), 5),
    ((2, 2, 1), 5),
    ((2, 2, 2), 5),
    ((2, 2, 1), 5),
    ((2, 1, 1), 2),
    ((2, 1, 2), 3),
    ((2, 2, 2), 5),
    ((2, 1, 1), 2),
    ((2, 2, 2), 5),
    ((2, 1, 1), 2),
    ((2, 2, 2), 5),
    ((2, 2, 1), 5),
    ((1, 2, 1), 4),
    ((2, 1, 1), 2),
    ((2, 2, 1), 5),
    ((1, 1, 1), 1),
]

sample0 = (2, 1, 1)

samples = [
    ((2, 2, 1), 4),
    ((1, 1, 2), 2),
    ((2, 2, 1), 4),
    ((1, 2, 1), 2),
    ((1, 1, 2), 2),
    ((1, 2, 1), 2),
    ((2, 1, 2), 4),
    ((2, 2, 2), 5),
    ((1, 1, 1), 1),
    ((1, 2, 2), 4),
    ((2, 1, 2), 4),
    ((1, 2, 2), 4),
    ((1, 1, 2), 2),
    ((1, 2, 2), 4),
    ((1, 1, 2), 2),
    ((1, 1, 2), 2),
    ((1, 2, 2), 4),
    ((1, 1, 2), 2),
    ((2, 1, 1), 2),
    ((2, 1, 2), 4),
    ((2, 2, 1), 4),
    ((1, 1, 2), 2),
    ((1, 1, 1), 1),
    ((2, 1, 2), 4),
    ((2, 2, 2), 5),
    ((2, 1, 2), 4),
    ((2, 1, 1), 2),
    ((1, 1, 1), 1),
    ((2, 2, 2), 5),
    ((1, 2, 2), 4),
]

sample = (1, 1, 2)

def train(samples):
    classes, freq = defaultdict(lambda:0), defaultdict(lambda:0)
    for feats, label in samples:
        classes[label] += 1                 # count classes frequencies
        for feat in feats:
            freq[label, feat] += 1          # count features frequencies

    for label, feat in freq:                # normalize features frequencies
        freq[label, feat] /= classes[label]
    for c in classes:                       # normalize classes frequencies
        classes[c] /= len(samples)

    return classes, freq                    # return P(C) and P(O|C)


def classify(classifier, feats):
    classes, prob = classifier
    return min(classes.keys(),              # calculate argmin(-log(C|O))
        key = lambda cl: -log(classes[cl]) + \
            sum(-log(prob.get((cl,feat), 10**(-7))) for feat in feats))


if __name__ == '__main__':
    classifier = train(samples)
    print(classify(classifier, sample))
