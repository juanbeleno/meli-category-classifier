#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocess the data.

Created on Mon Sep 16 22:23:17 2019
@author: Juan Beleño
"""


def preprocess_text(text: str, lowercase_flag: bool = False) -> str:
    """Remove unwanted characters"""
    clean_text = text.strip()
    if lowercase_flag:
        clean_text = clean_text.lower()

    return clean_text
