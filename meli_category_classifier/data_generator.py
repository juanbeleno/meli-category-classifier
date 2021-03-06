#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data generator from a big CSV file.

Created on Wed Sep 18 22:10:39 2019
@author: Juan Beleño
"""
import csv
import numpy as np

from tensorflow.python.keras.utils.data_utils import Sequence


class DataGenerator(Sequence):

    def __init__(self, dataset_filepath, category_map, data_size, config):
        self.dataset_filepath = dataset_filepath
        self.category_map = category_map
        self.data_size = data_size
        self.config = config
        self.data_reader = csv.reader(open(self.dataset_filepath),
                                      delimiter=',')

    def __len__(self):
        return int(np.ceil(self.data_size / self.config.batch_size))

    def __getitem__(self, idx):
        tokens = []
        lang = []
        outputs = []
        # Get the csv data for row = idx
        for index in range(self.config.batch_size):
            data = next(self.data_reader)
            random_tokens = self.apply_token_dropout(data[:self.config.max_sequence_length],
                                                     self.config.word_dropout)
            tokens.append(random_tokens)
            lang.append(data[self.config.max_sequence_length])

            category = int(data[self.config.max_sequence_length + 1])
            output = np.zeros(len(self.category_map))
            output[category] = 1
            outputs.append(output)

        inputs = {
            'tokens': np.array(tokens),
            'lang': np.array(lang)
        }

        outputs = np.array(outputs)

        return inputs, outputs

    def apply_token_dropout(self, tokens, dropout_rate):
        '''Randomly deletes tokens in the list of tokens'''
        random_tokens = [x for x in tokens if np.random.rand() > dropout_rate]
        random_tokens.extend([0] * (len(tokens) - len(random_tokens)))
        return random_tokens
