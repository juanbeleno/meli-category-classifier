#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modify the train dataset to bolster samples of categories where my model fails.

Created on Wed Sep 25 20:46:57 2019
@author: Juan Beleño
"""
import csv
import numpy as np

from .config import MeliClassifierConfig
from .dataset import load_category_map
from .files import MeliClassifierFiles
from .model import meli_model
from bpemb import BPEmb
from typing import Union


def save_bad_classification_weights(
        config: Union[str, MeliClassifierConfig] = MeliClassifierConfig()
) -> None:
    '''Modify weights based on the accuracy of the model per category'''
    if isinstance(config, str):
        config = MeliClassifierConfig.from_yaml(config)
    config.pretrained_classifier = True
    files = MeliClassifierFiles(config)

    category_map = load_category_map()

    val_file = csv.reader(open(files.test_dataset), delimiter=',')
    for row in val_file:

    model = meli_model(config)

    es_config = MeliClassifierConfig(lang='es')
    es_config.pretrained_classifier = True
    es_files = MeliClassifierFiles(es_config)

    pt_config = MeliClassifierConfig(lang='pt')
    pt_config.pretrained_classifier = True
    pt_files = MeliClassifierFiles(pt_config)

    es_bpemb = BPEmb(lang='es', vs=es_config.max_features,
                     dim=es_config.embed_size)
    pt_bpemb = BPEmb(lang='pt', vs=pt_config.max_features,
                     dim=pt_config.embed_size)

    category_map = load_category_map()
    inverse_category_map = {}
    for key in category_map:
        inverse_category_map[category_map[key]] = key
    

    es_model = meli_model(es_config)
    pt_model = meli_model(pt_config)

    with open(es_files.result_dataset, 'a') as result_file:
        result_file.write('id,category\n')

    input_file = csv.reader(open(es_files.result_input_dataset), delimiter=',')
    # Ignore header
    next(input_file, None)

    for row in input_file:
        if row[2] == 'spanish':
            tokens = es_bpemb.encode_ids(row[1])
            tokens.extend([0] * (es_config.max_sequence_length - len(tokens)))
            features = {
                'tokens': np.array([tokens]),
                'lang': np.array([lang_map[row[2]]])
            }
            prediction = es_model.predict(features)
            prediction_index = prediction.argmax(axis=1)[0]
            with open(es_files.result_dataset, 'a') as result_file:
                result_file.write('{0},{1}\n'.format(row[0],
                                  inverse_category_map[prediction_index]))
        else:
            tokens = pt_bpemb.encode_ids(row[1])
            tokens.extend([0] * (pt_config.max_sequence_length - len(tokens)))
            features = {
                'tokens': np.array([tokens]),
                'lang': np.array([lang_map[row[2]]])
            }
            prediction = pt_model.predict(features)
            prediction_index = prediction.argmax(axis=1)[0]
            with open(pt_files.result_dataset, 'a') as result_file:
                result_file.write('{0},{1}\n'.format(row[0],
                                  inverse_category_map[prediction_index]))

