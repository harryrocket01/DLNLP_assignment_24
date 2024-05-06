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
    class: DataProcessing

    Contains all the classes and tools inoder to perform dataprocessing
    for text based data for NLP, Deep Learning Models.

    args:
        none
    methods:
        __init__(self):
            Initializes the class attributes.

        sentence_processing()
        unicode_to_ascii()
        tokenize()
        load_dataset():
        call_train_val()
        call_test()

    Example:
        DataProcessing()
    """

    def __init__(self):
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None
        self.max_seq = None

    def sentence_processing(self, sentence: str):
        """
        method: sentence_processing

        Process input string to a uniform, lowercase
        ascii format.

        Example - Hello There What IS your Name.

        <hello there what is your name.>


        Note -  < = START OF SEQUENCE, > = END OF SEQUENCE
        args:
            sentence (str): string input to convert
        return:
            sentence (str): formatted string
        Example:
            DataProcessing().sentence_processing("Hello There")
        """
        sentence = sentence.lower().strip()
        sentence = self.unicode_to_ascii(sentence)

        sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
        sentence = re.sub(r'[" "]+', " ", sentence)
        sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
        sentence = sentence.strip()

        # SOS = <, EOS = >
        sentence = "<" + sentence + ">"
        return sentence

    def unicode_to_ascii(self, sentence):
        """
        method: unicode_to_ascii

        Converts base unicode to standard ascii.

        args:
            setence (): input to convert
        return:
            converted (str): converted string to ascii
        Example:
            DataProcessing().unicode_to_ascii("General Kenobi")
        """
        return "".join(
            converted
            for converted in unicodedata.normalize("NFD", sentence)
            if unicodedata.category(converted) != "Mn"
        )

    def tokenize(self, inp_lang, targ_lang):
        """
        method: train

        Tokanises the inpiutted text at a character level.

        args:
            inp_lang(arr): input text to convert
            targ_lang(arr): target text to convert
        return:
            tensor: Tensor of input language
            lang_tokenizer: Tensorflow Text Tokaniser instance for input
            tensor2: Tensor of target language
            lang_tokenizer: Tensorflow Text Tokaniser instance for target
        Example:
            DataProcessing().tokenize(Source, Target)
        """
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

    def load_dataset(
        self, num_examples: int = 10000, root: str = ".\Dataset\Misspelling_Corpus.csv"
    ):
        """
        method: load_dataset

        Loads target dataset, and splits in to training and test

        args:
            num_examples (int): number of lines to load from file
            root (str): location of CSV to load
        return:
            input_tensor: Tensor of input language
            lang_tokenizer: Tensorflow Text Tokaniser instance for input
            target_tensor: Tensor of target language

        Example:
            DataProcessing().tokenize(10000, ".\Dataset\Misspelling_Corpus.csv")

        """
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
        num_examples: int,
        buffer_size: int,
        batch_size: int,
        root: str = ".\Dataset\Misspelling_Corpus.csv",
    ):
        """
        method: call_train_val

        Creates training and validation data from selected root.

        args:
            num_examples (int): number of examples to load
            buffer_size (int): size of the buffer
            batch_size (int): model batch size
            root (str): location to load from

        return:
            train_dataset: zipped pairs of training data
            val_dataset: zipped pairs of validation data
            inp_lang_tokenizer: Tensorflow Text Tokaniser instance for input
            targ_lang_tokenizer: Tensorflow Text Tokaniser instance for input

        Example:
            call_train_val().tokenize(10000, 100, 64, ".\Dataset\Misspelling_Corpus.csv")

        """
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
        buffer_size: int,
        batch_size: int,
        inp_lang_tokenizer,
        targ_lang_tokenizer,
        root=".\Dataset\Test_Set.csv",
    ):
        """
        method: call_test

        Creates test data from selected root.

        args:
            num_examples (int): number of examples to load
            batch_size (int): model batch size
            root (str): location to load from
            inp_lang_tokenizer: Tensorflow Text Tokaniser instance for input
            targ_lang_tokenizer: Tensorflow Text Tokaniser instance for input

        return:
            test_dataset: zipped pairs of test data

        Example:
            call_train_val().test_dataset()
        """
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
        """
        method: convert_sequences_to_tokens

        QOL function that convertes a given sequence using a provided tokenizer

        args:
            sequences: sentneces to convert
            tokenizer: Tensorflow Text instance to use to covnert.
        return:
            Tensor of tokanised sentence
        Example:
            call_train_val().convert_sequences_to_tokens()

        """
        return tokenizer.sequences_to_texts(sequences)
