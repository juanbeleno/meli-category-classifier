#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modify the train dataset to bolster samples of categories where my model fails.

Created on Wed Sep 25 20:46:57 2019
@author: Juan Beleño
"""
import csv
import numpy as np
import pandas as pd

from .config import MeliClassifierConfig
from .dataset import load_category_map
from .files import MeliClassifierFiles
from .model import meli_model
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
    inverse_category_map = {}
    for key in category_map:
        inverse_category_map[category_map[key]] = key

    val_file = csv.reader(open(files.test_dataset), delimiter=',')
    tokens = []
    lang = []
    true_categories = []
    for row in val_file:
        tokens.append(row[:config.max_sequence_length])
        lang.append(row[config.max_sequence_length])
        true_categories.append(int(row[config.max_sequence_length + 1]))

    features = {
        'tokens': np.array(tokens),
        'lang': np.array(lang)
    }

    model = meli_model(config)
    predictions = model.predict(features)
    pred_categories = predictions.argmax(axis=1)

    good_results = {}
    bad_results = {}
    for idx in range(len(true_categories)):
        category = true_categories[idx]
        if category == pred_categories[idx]:
            good_results[category] = good_results.get(category, 0) + 1
        else:
            bad_results[category] = bad_results.get(category, 0) + 1

    results = []
    for key in inverse_category_map:
        denominator = good_results.get(key, 0) + bad_results.get(key, 0)
        accuracy = 1.0
        if denominator > 0:
            accuracy = good_results.get(key, 0) * 1.0 / denominator
        results.append({'category': key, 'accuracy': accuracy})

    results = pd.DataFrame(results)
