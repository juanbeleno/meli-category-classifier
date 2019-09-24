#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hanlding files.

Created on Wed Sep 18 20:46:23 2019
@author: Juan Beleño
"""
from typing import Union
from .config import MeliClassifierConfig
from .__version__ import __version__


class MeliClassifierFiles():
    _raw_data_filename: str = 'train.csv'
    _train_filename: str = '{}_train.csv'
    _test_filename: str = '{}_test.csv'
    _category_map_filename: str = 'category_map.json'
    _es_model_filename: str = f'es-meli-category-classifier-{__version__}.h5'
    _es_model_checkpoint: str = f'es-meli-category-classifier-{__version__}.hdf5'
    _pt_model_filename: str = f'pt-meli-category-classifier-{__version__}.h5'
    _pt_model_checkpoint: str = f'pt-meli-category-classifier-{__version__}.hdf5'

    def __init__(self,
                 config: Union[str, MeliClassifierConfig] = MeliClassifierConfig()):
        if isinstance(config, str):
            config = MeliClassifierConfig.from_yaml(config)
        self._input_dir = config.input_directory_path
        self._processed_dir = config.processed_directory_path
        self._output_dir = config.output_directory_path
        self._checkpoints_dir = config.checkpoints_directory_path

        self._input_dir.mkdir(parents=True, exist_ok=True)
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.lang = config.lang

    @property
    def raw_dataset(self) -> str:
        return str(self._input_dir/self._raw_data_filename)

    @property
    def train_dataset(self) -> str:
        return str(self._processed_dir/self._train_filename.format(self.lang))

    @property
    def test_dataset(self) -> str:
        return str(self._processed_dir/self._test_filename.format(self.lang))

    @property
    def model_weights(self) -> str:
        if self.lang == 'es':
            return str(self._output_dir/self._es_model_filename)
        return str(self._output_dir/self._pt_model_filename)

    @property
    def category_map(self) -> str:
        return str(self._processed_dir/self._category_map_filename)

    @property
    def model_checkpoint(self) -> str:
        if self.lang == 'es':
            return str(self._checkpoints_dir/self._es_model_checkpoint)
        return str(self._checkpoints_dir/self._pt_model_checkpoint)
