#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 22:29:46 2019
@author: Juan Beleño
"""
import fire

from .dataset import save_category_map, save_class_weights, split_train_test_data
from .predict import predict_results
from .train import train_model


def main():
    """Expose CLI functions."""
    fire.Fire({
        'save-category-map': save_category_map,
        'save-class-weights': save_class_weights,
        'split-data': split_train_test_data,
        'train-model': train_model,
        'predict-results': predict_results
    })
