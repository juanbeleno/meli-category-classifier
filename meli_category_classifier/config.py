#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file.

Created on Mon Sep 16 21:41:54 2019
@author: Juan Beleño
"""
import os
import yaml

from pathlib import Path


class MeliClassifierConfig:
    # Preprocessing
    # 16K or 32K subword tokens are good numbers given the size of the dataset.
    # 2017 - Stronger Baselines for Trustable Results in Neural Machine
    # Translation
    vocab_size: int = 2**15
    max_sequence_length: int = 120
    max_features: int = 320000
    embed_size: int = 300
    single_feature_size: int = 1
    data_size: int = 20000000
    test_size: int = 0.05

    # Model
    num_classes: int = 1588
    pretrained_classifier: bool = False
    dropout_rate: float = 0.5
    lstm_hidden_size: int = 16

    # Training
    verbose: int = 1
    n_epochs: int = 1
    batch_size: int = 1024
    num_training_samples: int = 18999195
    num_validation_samples: int = 1000805

    # Adam optimizer
    learning_rate: float = 1e-3
    beta_1: float = 0
    beta_2: float = 0.98
    epsilon: float = 1e-8

    # Fixed
    main_directory:str = os.path.join(os.path.dirname(os.path.realpath(__file__)))
    input_directory: str = os.path.join(main_directory, 'assets', 'inputs')
    processed_directory: str = os.path.join(main_directory, 'assets', 'processed')
    output_directory: str = os.path.join(main_directory, 'assets', 'outputs')
    checkpoints_directory: str = os.path.join(main_directory, 'assets', 'checkpoints')

    def __init__(self, pretrained_classifier=False):
        self.pretrained_classifier = pretrained_classifier

    @classmethod
    def from_yaml(cls, path: str):
        """Load overrides from a YAML config file."""
        with open(path) as configfile:
            configdict = yaml.safe_load(configfile)
        return cls(**configdict)

    @property
    def input_directory_path(self) -> Path:
        path_string = os.environ.get(
            'INPUT_DIRECTORY', self.input_directory)
        return Path(path_string)

    @property
    def processed_directory_path(self) -> Path:
        path_string = os.environ.get(
            'PROCESSED_DIRECTORY', self.processed_directory)
        return Path(path_string)

    @property
    def output_directory_path(self) -> Path:
        path_string = os.environ.get(
            'OUTPUT_DIRECTORY', self.output_directory)
        return Path(path_string)

    @property
    def checkpoints_directory_path(self) -> Path:
        path_string = os.environ.get(
            'CHECKPOINT_DIRECTORY', self.checkpoints_directory)
        return Path(path_string)
