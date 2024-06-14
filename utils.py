"""
Function of 'get_pcfg' and 'read_grammar' is originated from https://github.com/SiyuanQi-zz/generalized-earley-parser/blob/master/src/python/parser/grammarutils.py.
Function of 'read_mapping_dict' is originated from https://github.com/gongda0e/FUTR/blob/main/utils.py.
"""

import collections
import os
import time
import itertools

import numpy as np
import nltk

import functools


def get_pcfg(rules, index=False, mapping=None):
    root_rules = list()
    non_terminal_rules = list()
    grammar_rules = list()
    import pdb
    for rule in rules:
        tokens = rule.split()
        for i in range(len(tokens)):
            token = tokens[i]
            if token[0] == 'E':
                tokens[i] = tokens[i].replace('E', 'OR')
            elif token[0] == 'P':
                tokens[i] = tokens[i].replace('P', 'AND')
            elif index and mapping and token[0] == "'":
                tokens[i] = "'{}'".format(mapping[token.strip("'")])
            elif token[0] == "I":
                pass
        rule = ' '.join(tokens)

        if rule.startswith('S'):
            root_rules.append(rule)
        else:
            non_terminal_rules.append(rule)

    for k, v in collections.Counter(root_rules).items():
        grammar_rules.append(k + ' [{}]'.format(float(v) / len(root_rules)))
    grammar_rules.extend(non_terminal_rules)
    return grammar_rules


def read_grammar(filename, index=False, mapping=None, insert=True):
    with open(filename) as f:
        rules = [rule.strip() for rule in f.readlines()]
        if insert:
            rules.insert(0, 'GAMMA -> S [1.0]')
        grammar_rules = get_pcfg(rules, index, mapping)
        grammar = nltk.PCFG.fromstring(grammar_rules)
    return grammar

def read_mapping_dict(file_path):
    # github.com/yabufarha/anticipating-activities
    '''This function read action index from the txt file'''
    file_ptr = open(file_path, 'r')
    actions = file_ptr.read().split('\n')[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    return actions_dict

