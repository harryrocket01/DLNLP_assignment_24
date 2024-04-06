import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import pandas as pd
import os
import io
import time


class DataProcessing:
    def __init__(self):
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None

    ## Step 1 and Step 2
    def sentence_processing(self, sentence):
        sentence = sentence.lower().strip()
        sentence = self.unicode_to_ascii(sentence)

        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
        sentence = sentence.strip()

        # SOS = <, EOS = >
        sentence = "<" + sentence + ">"
        return sentence

    def unicode_to_ascii(self, s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    def tokenize(self, inp_lang, targ_lang):
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters="", oov_token="#", char_level=True
        )
        lang_tokenizer.fit_on_texts(targ_lang)

        tensor = lang_tokenizer.texts_to_sequences(inp_lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")

        tensor2 = lang_tokenizer.texts_to_sequences(targ_lang)
        tensor2 = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding="post")

        return tensor, lang_tokenizer, tensor2, lang_tokenizer

    def load_dataset(self, num_examples=10000, root=".\Dataset\Misspelling_Corpus.csv"):
        df = pd.read_csv(root, nrows=num_examples)
        df["Original"] = df["Original"].apply(self.sentence_processing)
        df["Misspelt"] = df["Misspelt"].apply(self.sentence_processing)

        targ_lang = df["Original"].values

        inp_lang = df["Misspelt"].values

        input_tensor, inp_lang_tokenizer, target_tensor, targ_lang_tokenizer = (
            self.tokenize(inp_lang, targ_lang)
        )

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    def call(self, num_examples, buffer_size, batch_size):

        (
            input_tensor,
            target_tensor,
            self.inp_lang_tokenizer,
            self.targ_lang_tokenizer,
        ) = self.load_dataset(num_examples)

        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = (
            train_test_split(input_tensor, target_tensor, test_size=0.2)
        )

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor_train, target_tensor_train)
        )
        train_dataset = train_dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=True
        )

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor_val, target_tensor_val)
        )
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True)

        return (
            train_dataset,
            val_dataset,
            self.inp_lang_tokenizer,
            self.targ_lang_tokenizer,
        )
