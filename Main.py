# Databricks notebook source
# %pip install gensim
# %pip install python-Levenshtein
# %pip install openpyxl
# %pip install pytz

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import gensim
import pandas as pd
import os
import Levenshtein
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql import Window
import requests
import re

from pyspark.sql import functions as F
from pyspark.sql.types import StringType, IntegerType, ArrayType
from pyspark.broadcast import Broadcast

import utils_model as fx
from utils_model import stopwords, special_letters

from datetime import datetime
import pytz
import time

from config import input_file, output_file, ratio_degree_level, ratio_speciality, match_score


# COMMAND ----------

model_path = "model/trained_model.model.model"

# COMMAND ----------

# Loading master tables
input_master_clean_words = pd.read_excel('master file/bag_of_keywords.xlsx', sheet_name = 0).drop_duplicates(subset=['clean_word'])
input_master_degrees = pd.read_excel('master file/bag_of_keywords.xlsx', sheet_name = 1)

# COMMAND ----------

# Loading main data
input_df = pd.read_csv(f'input/{input_file}', sep=',', on_bad_lines='skip', encoding="UTF-8")


# COMMAND ----------

spark = SparkSession.builder.appName("NLP app").getOrCreate()

# COMMAND ----------

# Creating the dfs
df =  spark.createDataFrame(input_df)
master_clean_words = spark.createDataFrame(input_master_clean_words)
master_degrees =  spark.createDataFrame(input_master_degrees)

# COMMAND ----------

# Cleanning the master degrees from stopwords, special characters, special letters etc
master_degrees = fx.clean_text_column(
    master_degrees,
    "degree_name",
    stopwords,
    special_letters,
    "[^a-z0-9\\s]"
)

# COMMAND ----------

df = fx.clean_text_column(
    df,
    "name",
    stopwords,
    special_letters,
    "[^a-z0-9\\s]"
)

# COMMAND ----------

# Pivoting the column into seaparate words
master_degrees_words = fx.master_degrees_prepare(master_degrees)

# COMMAND ----------

# creating bags of words lists for degree list and speciality 

target_specialities = master_degrees_words.dropDuplicates(['words']).select('words')
target_types = master_clean_words.select('clean_word')

target_specialities =  [row[0] for row in target_specialities.collect()]
target_types =  [row[0] for row in target_types.collect()]

target_specialities

# COMMAND ----------

""" Degree level clasification """
# correcting the spell of the column, storing the affected words original_tokens_matched and the corrected new word in matched_words
# Including the model which detected the word in model_sources where: 1 = word2vec, 2 = levenshtein, 3 = levenshtein second layer
# Example: if the record contains the word Masters it be replaced by Master to later match it with the degree level Master
df = fx.apply_multiple_spelling_corrections(
    df,
    "name",
    target_types,
    stopwords,
    special_letters,
    model_path,
    ratio_degree_level,
    "|"
)

df.show(50)

# COMMAND ----------

COLS_RENAME_TARGET_TYPES = {
    ('model_sources', 'model_source_types'),
    ('name_corrected', 'name')
}

COLS_RENAME_TARGET_SPECIALITY = {
    ('model_sources', 'model_source_specialities'),
    ('name_corrected', 'name'),
    ('matched_words', 'matched_speciality')
}

# COMMAND ----------

# Keeping the first word in matched type in order to join later with matched_clean_words to get the degree level
# This could be improved somehow, but for now this is the current method
df = df.withColumn(
    'matched_type',
    F.split(F.col("matched_words"), "\|")[0]
).drop('name','original_tokens_matched','matched_words').withColumnsRenamed(
    {old_name: new_name for old_name, new_name in COLS_RENAME_TARGET_TYPES}
)

df.show(40)

# COMMAND ----------

# Injecting the degree_level column
df = df.join(
    master_clean_words.withColumnRenamed('clean_word','matched_type'),
    on='matched_type',
    how='left'
).withColumnRenamed('type','degree_level')

# COMMAND ----------

""" Degree name clasification """
# correcting the spell of the column, storing the affected words original_tokens_matched and the corrected new word in matched_words
# This part is made to process the specialities
# Example: if the record contains the word Fnance it be replaced by Finance to later match it with the degree name Degree in Finance
df = fx.apply_multiple_spelling_corrections(
    df,
    "name",
    target_specialities,
    stopwords,
    special_letters,
    model_path,
    ratio_speciality,
    "|"
)

# COMMAND ----------

df = df.drop('name','original_tokens_matched').withColumnsRenamed(
    {old_name: new_name for old_name, new_name in COLS_RENAME_TARGET_SPECIALITY}
)

# COMMAND ----------

df = df.withColumn("id", F.col("id").cast(StringType()))
master_degrees = master_degrees.withColumn("id", F.col("id").cast(StringType()))

# COMMAND ----------

""" Degree name matching """
# The function tokenizes the matched words, corrjoins it with master_degrees to have all the available degree_names for every line, creates a matching score using the tokenized columns where it counts the matching words of both token columns, orders the candidates descending by the match_score and picks the winning candidate which has the rank 1, it considers only the candidates with the minimum match_score input
# If the match_score is minimum 2 words, then it does not take in count all the candidates with less than 2 words
df = fx.degree_clasification(df, master_degrees, match_score)

# COMMAND ----------

# Converting the array columns into string just to view them in the input and check the process
# This part could be excluded based on the need

df = df.withColumn(
    "education_tokens",
    F.concat(
        F.lit("['"),
        F.concat_ws("','", F.col("education_tokens")),
        F.lit("']")
    )
).repartition(8)

df = df.withColumn(
    "master_tokens",
    F.concat(
        F.lit("['"),
        F.concat_ws("','", F.col("master_tokens")),
        F.lit("']")
    )
).repartition(8)


# COMMAND ----------

df = df.select(
    'id',
    'name',
    'cluster',
    'degree_name',
    'degree_level',
    'matched_type',
    'education_tokens',
    'master_tokens',
    'match_score',
    'model_source_specialities',
    'model_source_types'
)

# COMMAND ----------

# Making sure only the degree_name with the minimum match_score are showing
df = df.withColumn(
    'degree_name',
    F.when(
        F.col('match_score') < match_score, F.lit(None)
    ).when(
        F.col('match_score').isNull(), F.lit(None)
    ).otherwise(F.col('degree_name') )
)

# COMMAND ----------

# Saving outpu, converting the df to pandas because Databricks free tier does not let to directly save to a file from Spark df
df = df.toPandas()

# COMMAND ----------

df.to_csv(f"output/{output_file}", index=False, encoding="utf-8-sig", quotechar='"', quoting=1)