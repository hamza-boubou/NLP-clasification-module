# Databricks notebook source
# %pip install gensim
# %pip install python-Levenshtein
# %pip install openpyxl

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import gensim
import pandas as pd
import os
import Levenshtein
import requests
import re
import glob

import utils_model as fx
from utils_model import stopwords, special_letters

# COMMAND ----------

folder = 'training data'

df = fx.combine_csv_files(folder, ['id', 'name', 'cluster'])

df

# COMMAND ----------

# Apply the function to the 'text_column'
df['name'] = df['name'].apply(fx.clean_text)

# COMMAND ----------

df.shape

# COMMAND ----------

review_text = df.name.apply(gensim.utils.simple_preprocess)

review_text

# COMMAND ----------

review_text.loc[0]

# COMMAND ----------

""" initialize model """

model = gensim.models.Word2Vec(
    window=5,
    min_count=2,
    workers=4,
)

# COMMAND ----------

model.build_vocab(review_text, progress_per=1000)

# COMMAND ----------

model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)

# COMMAND ----------

model.save("model/trained_model.model")

# COMMAND ----------

model.wv.most_similar("accounting")