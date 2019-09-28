#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model.

Source:
- https://paperswithcode.com/paper/revisiting-lstm-networks-for-semi-supervised

Created on Wed Sep 18 23:14:11 2019
@author: Juan Beleño
"""
import tensorflow as tf

from .config import MeliClassifierConfig
from .files import MeliClassifierFiles
from bpemb import BPEmb
from tensorflow.python.keras.layers import (
    Input, Dense, Dropout, GlobalMaxPool1D, concatenate, LSTM, Bidirectional,
    Embedding
)
from tensorflow.python.keras.models import Model
from typing import Union


def meli_model(
        config: Union[str, MeliClassifierConfig] = MeliClassifierConfig()
) -> Model:
    '''First model for Mercado Libre Challenge'''
    if isinstance(config, str):
        config = MeliClassifierConfig.from_yaml(config)

    lang_bpemb = BPEmb(lang=config.lang, vs=config.max_features,
                       dim=config.embed_size)

    input_tokens = Input(shape=(config.max_sequence_length,), name='tokens')
    input_lang = Input(shape=(config.single_feature_size,), name='lang')

    x = Embedding(config.max_features, config.embed_size,
                  weights=[lang_bpemb.vectors])(input_tokens)
    x = Bidirectional(LSTM(config.lstm_hidden_size, return_sequences=True,
             dropout=config.dropout_rate))(x)
    x = GlobalMaxPool1D()(x)
    x = concatenate([x, input_lang])
    x = Dense(config.num_classes, activation="linear")(x)
    x = Dropout(config.dropout_rate)(x)
    x = Dense(config.num_classes, activation="softmax")(x)
    model = Model(inputs=[input_tokens, input_lang], outputs=x)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(config.learning_rate, config.beta_1,
                                           config.beta_2, config.epsilon),
        metrics=['accuracy'],
        weighted_metrics=['accuracy']
    )

    if config.pretrained_classifier:
        files = MeliClassifierFiles(config)
        model.load_weights(files.model_weights)

    return model
