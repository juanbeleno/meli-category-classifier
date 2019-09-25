#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the model.

Created on Fri Sep 20 21:26:44 2019
@author: Juan Beleño
"""
from .config import MeliClassifierConfig
from .data_generator import DataGenerator
from .dataset import load_category_map
from .files import MeliClassifierFiles
from .model import meli_model
from bpemb import BPEmb
from typing import Union
from tensorflow.python.keras.callbacks import (
    ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.python.keras.models import load_model


def train_model(
    config: Union[str, MeliClassifierConfig] = MeliClassifierConfig()
) -> None:
    """Train the model and save classifier and feature weights."""
    if isinstance(config, str):
        config = MeliClassifierConfig.from_yaml(config)
    config.pretrained_classifier = True
    files = MeliClassifierFiles(config)
    category_map = load_category_map()

    training_generator = DataGenerator(
        files.train_dataset, category_map, config.num_training_samples, config)

    validation_generator = DataGenerator(files.test_dataset, category_map,
                                         config.num_validation_samples, config)

    model = meli_model(config)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                  min_lr=0.00005)
    model_checkpoint = ModelCheckpoint(filepath=files.model_checkpoint,
                                       monitor='val_loss', save_best_only=True)

    model.fit_generator(
        generator=training_generator,
        steps_per_epoch=(config.num_training_samples // config.batch_size),
        epochs=config.n_epochs,
        validation_data=validation_generator,
        validation_steps=(config.num_validation_samples // config.batch_size),
        verbose=config.verbose,
        callbacks=[reduce_lr, model_checkpoint],
        use_multiprocessing=True)

    model = load_model(files.model_checkpoint)
    model.save_weights(files.model_weights, overwrite=True)
    return 'OK'
