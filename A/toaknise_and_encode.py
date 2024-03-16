import numpy as np
import re
import pandas as pd

import subprocess
import sys


subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "--upgrade", "tensorflow-addons"]
)


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


class TokaniseAndEncode:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary

    def process(self, input_text):
        max_length = max(len(sentence) for sentence in input_text)

        padded_input_text = [sentence.ljust(max_length) for sentence in input_text]

        encoded_text = []
        for line in padded_input_text:
            tokenized_text = self.tokenise(line)
            print(tokenized_text)
            marked_text = self.position_marker(tokenized_text)
            print(marked_text)
            encoded_sentence = self.encode(marked_text)
            print(encoded_sentence)
            encoded_text.append(encoded_sentence)
            print(encoded_text)

        print(self.vocabulary)
        return encoded_text

    def tokenise(self, input_text):
        "Character-based tokenization"
        tokens = list(input_text)
        return tokens

    def position_marker(self, tokens):
        "Inserts SOS, EOS, and punctuation tokens within the text."
        # Assuming we mark the beginning and end of the sentence with SOS and EOS tokens respectively
        marked_tokens = ["<SOS>"] + tokens + ["<EOS>"]
        return marked_tokens

    def encode(self, tokens):
        "Converts tokenized text to a one-hot encoded vector"

        encoded_sentence = []
        for token in tokens:
            encoding = np.zeros(len(self.vocabulary))
            encoding[self.vocabulary[token]] = 1
            encoded_sentence.append(encoding)

        return encoded_sentence


test = ["Hello there I AM HARRY", "Lets Try This Out.", "What is that, its the unknown"]
vocab = Vocabularay().generate_vocab(test)
print(vocab)
inst = TokaniseAndEncode(vocab)
inst.process(test)
