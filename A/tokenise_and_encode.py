import os
import subprocess
import sys

# subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])

import numpy as np
import re
import pandas as pd
import tensorflow as tf
#import tensorflow_text as tf_text
from sklearn.model_selection import train_test_split

np.random.seed(1234)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


class Vocabularay:
    def __init__(self):
        pass

    def generate_vocab(self, sentences, encoding_dict=None):
        if encoding_dict is None:
            encoding_dict = {"<SOS>": 0, "<EOS>": 1}

        vocabulary = set(encoding_dict.keys())

        for sentence in sentences:
            for char in sentence:
                if char not in vocabulary:
                    vocabulary.add(char)

        vocabulary = sorted(vocabulary)

        next_index = len(encoding_dict)
        for char in vocabulary:
            if char not in encoding_dict:
                encoding_dict[char] = next_index
                next_index += 1

        return encoding_dict


class TokeniseAndEncode:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.max_len = 100
        self.train_size = 200

    def set_limits(self, max_len=None, train_size=None):
        self.max_len = max_len if max_len is not None else self.max_len
        self.train_size = train_size if train_size is not None else self.train_size

    def process(self, input_text):
        max_length = max(len(sentence) for sentence in input_text)
        padded_input_text = [sentence.ljust(max_length) for sentence in input_text]

        encoded_text = []

        for line in padded_input_text:
            tokenized_text = self.tokenise(line)

            marked_text = self.position_marker(tokenized_text)

            encoded_sentence = self.encode(marked_text)

            encoded_text.append(encoded_sentence)

        return encoded_text

    def tokenise(self, input_text):
        "Character-based tokenization"
        tokens = list(input_text)
        return tokens

    def position_marker(self, tokens):
        "Inserts SOS, EOS, and punctuation tokens within the text."
        # assuming we mark the beginning and end of the sentence with SOS and EOS tokens respectively
        marked_tokens = ["<SOS>"] + tokens + ["<EOS>"]
        marked_tokens = np.array(marked_tokens)
        return marked_tokens

    def encode(self, tokens):
        "Converts tokenized text to a one-hot encoded vector"

        encoded_sentence = []
        for token in tokens:
            encoding = np.zeros(len(self.vocabulary))
            encoding[self.vocabulary[token]] = 1
            encoded_sentence.append(encoding)

        encoded_sentence = np.array(encoded_sentence)
        return encoded_sentence

    def convert_list_to_tensor(self, context, target, buffer=100000, batch_size=64):
        buffer = len(context)
        is_train = np.random.uniform(size=(len(target),)) < 0.8

        train_raw = (
            tf.data.Dataset.from_tensor_slices((context[is_train], target[is_train]))
            .shuffle(buffer)
            .batch(batch_size)
        )
        val_raw = (
            tf.data.Dataset.from_tensor_slices((context[~is_train], target[~is_train]))
            .shuffle(buffer)
            .batch(batch_size)
        )

        for example_context_strings, example_target_strings in train_raw.take(1):
            print(example_context_strings[:5])
            print()
            print(example_target_strings[:5])
            break

        max_vocab_size = 5000

        context_text_processor = tf.keras.layers.TextVectorization(
            standardize=self.tf_lower_and_split_punct,
            split="character",
            max_tokens=max_vocab_size,
            ragged=True,
        )

        context_text_processor.adapt(train_raw.map(lambda context, target: context))

        target_text_processor = tf.keras.layers.TextVectorization(
            standardize="lower_and_strip_punctuation", split="character", ragged=True
        )
        target_text_processor = tf.keras.layers.TextVectorization(
            standardize=self.tf_lower_and_split_punct,
            split="character",
            max_tokens=max_vocab_size,
            ragged=True,
        )
        target_text_processor.adapt(train_raw.map(lambda context, target: context))

        print(target_text_processor.get_vocabulary())
        print(context_text_processor.get_vocabulary())

        # context_vocab = context_text_processor.get_vocabulary()
        # target_vocab = target_text_processor.get_vocabulary()

        def inner_process_text(context, target):
            context = context_text_processor(context).to_tensor()
            target = target_text_processor(target)
            targ_in = target[:, :-1].to_tensor()
            targ_out = target[:, 1:].to_tensor()
            return (context, targ_in), targ_out

        train_ds = train_raw.map(inner_process_text, tf.data.AUTOTUNE)
        val_ds = val_raw.map(inner_process_text, tf.data.AUTOTUNE)

        return train_ds, val_ds, context_text_processor, target_text_processor

    def tf_lower_and_split_punct(self, text):

        text = tf_text.normalize_utf8(text, "NFKD")
        text = tf.strings.lower(text)
        text = tf.strings.regex_replace(text, "a-z.?!,-", "")
        text = tf.strings.regex_replace(text, ".?!,-", r" \0 ")
        text = tf.strings.strip(text)
        # [ is SOS and ] is EOS
        text = tf.strings.join(["[", text, "]"], separator="")
        return text

    def convert_to_tensor_format(self, input_arr, output_arr, buffer_size):

        input_arr = np.array(input_arr).astype(np.float32)
        output_arr = np.array(output_arr).astype(np.float32)

        input_train, input_val, output_train, output_val = train_test_split(
            input_arr, output_arr, test_size=0.2, random_state=42
        )

        input_train_tensor = tf.convert_to_tensor(input_train, dtype=tf.int32)
        output_train_tensor = tf.convert_to_tensor(output_train, dtype=tf.int32)
        input_val_tensor = tf.convert_to_tensor(input_val, dtype=tf.int32)
        output_val_tensor = tf.convert_to_tensor(output_val, dtype=tf.int32)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (input_train_tensor, output_train_tensor)
        )
        train_dataset = train_dataset.shuffle(buffer_size).batch(
            buffer_size, drop_remainder=True
        )

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (input_val_tensor, output_val_tensor)
        )
        val_dataset = val_dataset.batch(buffer_size, drop_remainder=True)

        return train_dataset, val_dataset, self.vocabulary
