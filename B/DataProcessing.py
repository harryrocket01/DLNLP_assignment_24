"""
The following code was written as the final project for 
ELEC0141 Deep Learning for Natural Language Processing

Author: Harry R J Softley-Graham
Date: Jan-May 2024

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

from sklearn.model_selection import train_test_split

import unicodedata
import re
import pandas as pd


class DataProcessing:
    """
    class: Encoder

    ~~ DESC ~~
    args:

    methods:

    Attributes:

    Example:

    """

    def __init__(self):
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None
        self.max_seq = None

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
        tensor = tf.keras.preprocessing.sequence.pad_sequences(
            tensor, padding="post", maxlen=self.max_seq
        )

        tensor2 = lang_tokenizer.texts_to_sequences(targ_lang)
        tensor2 = tf.keras.preprocessing.sequence.pad_sequences(
            tensor2, padding="post", maxlen=self.max_seq
        )

        return tensor, lang_tokenizer, tensor2, lang_tokenizer

    def load_dataset(self, num_examples=10000, root=".\Dataset\Misspelling_Corpus.csv"):
        df = pd.read_csv(root, nrows=num_examples)
        df["Original"] = df["Original"].apply(self.sentence_processing)
        df["Misspelt"] = df["Misspelt"].apply(self.sentence_processing)
        original_length = df["Original"].apply(len).max()
        misspelt_length = df["Misspelt"].apply(len).max()
        self.max_seq = max(original_length, misspelt_length) + 1

        targ_lang = df["Original"].values

        inp_lang = df["Misspelt"].values

        print("Maxlen", self.max_seq)

        input_tensor, inp_lang_tokenizer, target_tensor, targ_lang_tokenizer = (
            self.tokenize(inp_lang, targ_lang)
        )

        return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

    def call_train_val(
        self,
        num_examples,
        buffer_size,
        batch_size,
        root=".\Dataset\Misspelling_Corpus.csv",
    ):

        (
            input_tensor,
            target_tensor,
            self.inp_lang_tokenizer,
            self.targ_lang_tokenizer,
        ) = self.load_dataset(num_examples, root=root)

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

    def call_test(
        self,
        buffer_size,
        batch_size,
        inp_lang_tokenizer,
        targ_lang_tokenizer,
        root=".\Dataset\Test_Set.csv",
    ):

        df = pd.read_csv(root)
        df["Original"] = df["Original"].apply(self.sentence_processing)
        df["Misspelt"] = df["Misspelt"].apply(self.sentence_processing)

        targ_lang = df["Original"].values

        inp_lang = df["Misspelt"].values

        tensor = self.inp_lang_tokenizer.texts_to_sequences(inp_lang)
        tensor = tf.keras.preprocessing.sequence.pad_sequences(
            tensor, padding="post", maxlen=self.max_seq
        )

        tensor2 = self.targ_lang_tokenizer.texts_to_sequences(targ_lang)

        tensor2 = tf.keras.preprocessing.sequence.pad_sequences(
            tensor2, padding="post", maxlen=self.max_seq
        )

        test_dataset = tf.data.Dataset.from_tensor_slices((tensor, tensor2))
        test_dataset = test_dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=True
        )

        return test_dataset

    def convert_sequences_to_tokens(self, sequences, tokenizer):
        return tokenizer.sequences_to_texts(sequences)
