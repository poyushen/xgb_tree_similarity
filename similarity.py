#!/usr/bin/env python3
import numpy as np
import pickle
import math
import xgboost
from sys import argv, path
from anytree import Node, RenderTree

path.append('./util')
import feature_transformers

def dump(clf, x):
    x = np.loadtxt(x, delimiter=',', dtype=str)
    if len(x.shape) == 1:
        x = x.reshape(1, -1)
    clf = pickle.load(open(clf, 'rb'))

    xgb = clf.steps[1][1]
    ft = clf.steps[0][1]
    x = xgboost.DMatrix(ft.transform(x))
    leaves = xgb.get_booster().predict(x, pred_leaf = True)
    #print(xgb.get_booster.get_dump())

    ## save all the trees
    trees = []
    for i in xgb.get_booster().get_dump():
        tmp = []
        for node in i.split('\n'):
            score = float(node[node.find('=') + 1:]) if 'leaf' in node else 0
            tmp.append({'val': node[:node.find(':')].strip(), 'depth': node.count('\t'), 'score': score})
        trees.append(tmp[:-1])
    #print(tree[0])

    return trees, leaves

def tree_score(tree, leaf):
    #save the score for one sample in one tree
    x = next((index for (index, item) in enumerate(tree) if item['val'] == leaf), None)
    score = []
    for i in range(len(tree)):
        if i == x: score.append(tree[i]['score'])
        else: score.append(0)

    return score

def tree_distance(tree, leaf1, leaf2):
    #save the distance between two samples in one tree

    ## distance
    x1 = next((index for (index, item) in enumerate(tree) if item['val'] == leaf1), None)
    x2 = next((index for (index, item) in enumerate(tree) if item['val'] == leaf2), None)

    d1 = tree[x1]['depth']
    d2 = tree[x2]['depth']

    p1 = []
    p2 = []

    for i in range(x1, -1, -1):
        if tree[i]['depth'] == d1 and d1 >= 0:
            p1.append(tree[i]['val'])
            d1 -= 1
    for i in range(x2, -1, -1):
        if tree[i]['depth'] == d2 and d2 >= 0:
            p2.append(tree[i]['val'])
            d2 -= 1

    flag = 0
    for i in p1:
        for j in p2:
            if i==j:
                flag = 1
                distance = p1.index(i) + p2.index(j)
            if flag == 1: break

    return distance

def info_all(trees, sample1, sample2):
    score_s1 = []
    score_s2 = []
    distance = []

    for i in range(len(sample1)):
        score_s1.append(tree_score(trees[i], str(sample1[i])))
        score_s2.append(tree_score(trees[i], str(sample2[i])))

        distance.append(tree_distance(trees[i], str(sample1[i]), str(sample2[i])))

    return {'score': [score_s1, score_s2], 'distance': distance}

if __name__ == "__main__":
    config = {
        'models': list(argv[1].split(',')),
    }

    trees, samples = dump(argv[1], argv[2])
    score = info_all(trees, samples[1], samples[0])['score']
    distance = info_all(trees, samples[1], samples[0])['distance']
