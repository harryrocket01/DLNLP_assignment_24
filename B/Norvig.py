"""
The following code was written as the final project for 
ELEC0141 Deep Learning for Natural Language Processing

Author: Harry R J Softley-Graham
Date: Jan-May 2024

"""

import sys
import subprocess


from spellchecker import SpellChecker
import nltk
from ngram import NGram
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk

import os
from nltk import word_tokenize
import itertools
import pandas as pd


class Norvig:
    def __init__(self):
        self.N = 1
        if not nltk.corpus.words.words():
            nltk.download("words")
        self.corpus = set(nltk.corpus.words.words())

    def levenshtein_distance(self, s1, s2):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """

        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row = [i + 1]

            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))

            previous_row = current_row

        return previous_row[-1]

    def spellcheck(self, sentence):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """

        predicts = []

        for word in sentence.split():
            suggested_word = word
            min_distance = float("inf")

            for vocab_word in self.corpus:
                distance = self.levenshtein_distance(word, vocab_word)
                if distance < min_distance:
                    min_distance = distance
                    suggested_word = vocab_word

            predicts.append(suggested_word)

        corrected_sentence = " ".join(predicts) + "."
        return corrected_sentence

    def spellcheck_addon(self, sentence):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """

        spell = SpellChecker(distance=self.N)
        corrected_words = []
        misspelled = spell.unknown(sentence.split())

        for word in sentence.split():
            if word not in self.corpus:
                corrected_word = spell.correction(word)
                if corrected_word is not None:
                    corrected_words.append(corrected_word)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)

        corrected_sentence = " ".join(corrected_words)
        corrected_sentence += "."
        return corrected_sentence

    def test(self, csv_file, addon=True):
        df = pd.read_csv(csv_file)
        if addon:
            df["Corrected"] = df["Misspelt"].apply(self.spellcheck_addon)

        else:
            df["Corrected"] = df["Misspelt"].apply(self.spellcheck)

        accuracy, precision, recall, f1 = self.calculate_metrics(df)

        print(
            f"{csv_file} - accuracy:{accuracy} , precision:{precision} , recall_score:{recall} , f1:{f1}"
        )
        return df

    def live_correction(self):
        for x in range(0, 20):
            user_input = input().lower()
            self.spellcheck_addon(user_input)

    def calculate_metrics(self, df):
        original_sentences = df["Original"].tolist()
        corrected_sentences = df["Corrected"].tolist()

        max_length = max(
            len(sentence) for sentence in original_sentences + corrected_sentences
        )

        original_chars = [
            list(sentence.ljust(max_length)) for sentence in original_sentences
        ]
        corrected_chars = [
            list(sentence.ljust(max_length)) for sentence in corrected_sentences
        ]

        original_flat = [
            char for sentence_chars in original_chars for char in sentence_chars
        ]
        corrected_flat = [
            char for sentence_chars in corrected_chars for char in sentence_chars
        ]

        accuracy = accuracy_score(original_flat, corrected_flat)
        precision = precision_score(
            original_flat, corrected_flat, average="weighted", zero_division=1
        )
        recall = recall_score(
            original_flat, corrected_flat, average="weighted", zero_division=1
        )
        f1 = f1_score(
            original_flat, corrected_flat, average="weighted", zero_division=1
        )

        return accuracy, precision, recall, f1


if __name__ == "__main__":
    levenshtein = Norvig()
    levenshtein.test("Dataset/Test_Set.csv")
