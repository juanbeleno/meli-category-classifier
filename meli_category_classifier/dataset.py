#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file manages the creation, transformation and presentation of the input
data.

Created on Mon Sep 16 21:39:00 2019
@author: Juan Beleño
"""
import csv
import json
import numpy as np
from typing import Union
import sys

from .config import MeliClassifierConfig
from .files import MeliClassifierFiles
from bpemb import BPEmb


def save_category_map(
        config: Union[str, MeliClassifierConfig] = MeliClassifierConfig()):
    '''Store the category map from strings to numbers'''
    if isinstance(config, str):
        config = MeliClassifierConfig.from_yaml(config)
    files = MeliClassifierFiles(config)
    category_map = {}
    raw_file = csv.reader(open(files.raw_dataset), delimiter=',')
    # Ignore header
    next(raw_file, None)
    for line in raw_file:
        category_map[line[3]] = category_map.get(line[3], len(category_map))

    with open(files.category_map, 'w') as json_file:
        json.dump(category_map, json_file)


def load_category_map(
        config: Union[str, MeliClassifierConfig] = MeliClassifierConfig()):
    '''Get the category map from a json file'''
    if isinstance(config, str):
        config = MeliClassifierConfig.from_yaml(config)
    files = MeliClassifierFiles(config)
    with open(files.category_map) as json_file:
        category_map = json.load(json_file)
    return category_map


def save_class_weights(
        config: Union[str, MeliClassifierConfig] = MeliClassifierConfig()):
    '''Store the class weights'''
    if isinstance(config, str):
        config = MeliClassifierConfig.from_yaml(config)
    files = MeliClassifierFiles(config)
    category_map = load_category_map()

    class_weights = {}
    raw_file = csv.reader(open(files.raw_dataset), delimiter=',')
    # Ignore header
    next(raw_file, None)
    for line in raw_file:
        category = category_map[line[3]]
        class_weights[category] = class_weights.get(category, 0) + 1

    for key in class_weights:
        class_weights[key] = 1.0 / class_weights[key]

    with open(files.class_weights, 'w') as json_file:
        json.dump(class_weights, json_file)


def load_class_weights(
        config: Union[str, MeliClassifierConfig] = MeliClassifierConfig()):
    '''Get the class weights from a json file'''
    if isinstance(config, str):
        config = MeliClassifierConfig.from_yaml(config)
    files = MeliClassifierFiles(config)
    with open(files.class_weights) as json_file:
        class_weights = json.load(json_file)
    clean_class_weights = {}
    for key in class_weights:
        clean_class_weights[int(key)] = class_weights[key]
    return clean_class_weights


def split_train_test_data():
    '''Split the raw dataset into test and training datasets'''
    es_config = MeliClassifierConfig(lang='es')
    es_files = MeliClassifierFiles(es_config)

    pt_config = MeliClassifierConfig(lang='pt')
    pt_files = MeliClassifierFiles(pt_config)

    es_bpemb = BPEmb(lang='es', vs=es_config.max_features,
                       dim=es_config.embed_size)
    pt_bpemb = BPEmb(lang='pt', vs=pt_config.max_features,
                       dim=pt_config.embed_size)

    # Manual mapping for languages
    lang_map = {
        'spanish': 0,
        'portuguese': 1
    }
    category_map = load_category_map()

    raw_file = csv.reader(open(es_files.raw_dataset), delimiter=',')
    # Ignore header
    next(raw_file, None)
    es_max_sequence_length = 0
    pt_max_sequence_length = 0
    for row in raw_file:
        if row[2] == 'spanish':
            tokens = es_bpemb.encode_ids(row[0])
            if len(tokens) > es_max_sequence_length:
                es_max_sequence_length = len(tokens)
            tokens.extend([0] * (es_config.max_sequence_length - len(tokens)))
            tokens.append(lang_map[row[2]])
            tokens.append(category_map[row[3]])
            line = ','.join(map(str, tokens))

            if np.random.rand() > es_config.test_size:
                with open(es_files.train_dataset, 'a') as train_file:
                    train_file.write('{0}\n'.format(line))
            else:
                with open(es_files.test_dataset, 'a') as test_file:
                    test_file.write('{0}\n'.format(line))
        else:
            tokens = pt_bpemb.encode_ids(row[0])
            if len(tokens) > pt_max_sequence_length:
                pt_max_sequence_length = len(tokens)
            tokens.extend([0] * (pt_config.max_sequence_length - len(tokens)))
            tokens.append(lang_map[row[2]])
            tokens.append(category_map[row[3]])
            line = ','.join(map(str, tokens))

            if np.random.rand() > pt_config.test_size:
                with open(pt_files.train_dataset, 'a') as train_file:
                    train_file.write('{0}\n'.format(line))
            else:
                with open(pt_files.test_dataset, 'a') as test_file:
                    test_file.write('{0}\n'.format(line))
    print('ES MAX LENGTH: {}'.format(es_max_sequence_length), file=sys.stderr)
    print('PT MAX LENGTH: {}'.format(pt_max_sequence_length), file=sys.stderr)
