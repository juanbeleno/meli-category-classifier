#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis.

I use this file to understand the distribution of the data given many
approaches. This is a manual work, so some code is garbage. However, I get some
insights from the data. I use Spyder to show results in realtime, so
I don't store unnecesary information.

Note: I'm going to try to comment the insights obtained from the data, but
sometimes I forget to do it, so sorry :(

I'm going to handle a big csv file and my Mackbook has a limited RAM memory, so
I choose to iterate over the lines of the file to get the data analysis.

The train.csv.gz is compressed so I manually decompressed.

Created on Mon Sep 16 19:55:56 2019
@author: Juan Beleño
"""
import csv
import matplotlib.pyplot as plt


train_filename = 'assets/inputs/train.csv'

# According to the Mercado Libre description of the data (available in
# https://ml-challenge.mercadolibre.com/downloads on 16/09/2019),
# the training file has 4 columns: Title, Language, Label quality, and
# Category. However, I need to know the exact column names of the CSV file
# (if they even exist).
with open(train_filename) as train_file:
    head = [next(train_file) for x in range(50)]
# Fact #1: The actual name of the four columns are title, label_quality,
# language, and category


# There are 2 languages in the dataset but I need to know the distribution
lang_counter = {
   'spanish': 0,
   'portuguese': 0
}
train_file = csv.reader(open(train_filename), delimiter=',')
# Ignore header
next(train_file, None)
for line in train_file:
    lang_counter[line[2]] = lang_counter[line[2]] + 1
# Fact #2: There are 10,000,000 of item titles per language.


# I need to know the distribution of classes
class_counter = {}
train_file = csv.reader(open(train_filename), delimiter=',')
# Ignore header
next(train_file, None)
for line in train_file:
    class_counter[line[3]] = class_counter.get(line[3], 0) + 1
# Fact #3: There are 1588 classes
# Fact #4: The number of elements per class fluctuates between 109
# (HAMBURGER_FORMERS) and 35973 (PANTS)


# Plotting a histogram to see how is the distribution of classes
class_frequencies = list(class_counter.values())
plt.hist(class_frequencies, bins='auto')
# Fact #5: There are many categories with less than 5000 elements (Block #1).
# Between 5000 and 17500 (Block #2) in average there are less elements than in
# Block #1. From this point onwards the distribution seems like the second half
# of a normal distribution.


# I want to know the distribution of the number of characters in the dataset
char_counter = {}
train_file = csv.reader(open(train_filename), delimiter=',')
# Ignore header
next(train_file, None)
for line in train_file:
    char_counter[len(line[0])] = char_counter.get(len(line[0]), 0) + 1
# Fact #6: The number of characters in the dataset per element fluctuates
# between 3 and 120.


# Plotting a graphic to know how is the distribution of number of characters
num_char_x = list(char_counter.keys())
num_char_y = list(char_counter.values())
plt.bar(num_char_x, num_char_y)
# plt.bar(num_char_x[57:], num_char_y[57:])
# Fact #7: The number of characters per title grows exponentially until 60
# characters. From that point onwards, the distribution seems like a truncated
# exponential distribution.
