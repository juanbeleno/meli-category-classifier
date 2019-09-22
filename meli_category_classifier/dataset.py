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


def split_train_test_data(
        config: Union[str, MeliClassifierConfig] = MeliClassifierConfig()):
    '''Split the raw dataset into test and training datasets'''
    if isinstance(config, str):
        config = MeliClassifierConfig.from_yaml(config)
    files = MeliClassifierFiles(config)

    multibpemb = BPEmb(lang="multi", vs=config.max_features,
                       dim=config.embed_size)
    # Manual mapping for languages
    lang_map = {
        'spanish': 0,
        'portuguese': 1
    }
    category_map = load_category_map()

    raw_file = csv.reader(open(files.raw_dataset), delimiter=',')
    # Ignore header
    next(raw_file, None)
    for row in raw_file:
        tokens = multibpemb.encode_ids(row[0])
        tokens.extend([0] * (config.max_sequence_length - len(tokens)))
        tokens.append(lang_map[row[2]])
        tokens.append(category_map[row[3]])
        line = ','.join(map(str, tokens))

        if np.random.rand() > config.test_size:
            with open(files.train_dataset, 'a') as train_file:
                train_file.write('{0}\n'.format(line))
        else:
            with open(files.test_dataset, 'a') as test_file:
                test_file.write('{0}\n'.format(line))
