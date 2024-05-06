"""
The following code was written as the final project for 
ELEC0141 Deep Learning for Natural Language Processing

Author: Harry R J Softley-Graham
Date: Jan-May 2024

"""

import sys
import subprocess


subprocess.check_call([sys.executable, "-m", "pip", "install", "ngram"])

from ngram import NGram
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk

nltk.download("words")


class Levenshtein:
    """
    class: Levenshtein

    Implements the Levenshtein distance algorithm for spellchecking text.

    Attributes:
        N (int): N-gram parameter for Levenshtein distance calculation.
        corpus (set): Set of words from NLTK's corpus used for spellchecking.

    Methods:
        __init__():
            Initializes the Levenshtein object.

        spellcheck(sentence):
            Corrects misspelled words in a sentence.

        test(csv_file):
            Tests the spellcheck method on a dataset and calculates evaluation metrics.

        calculate_metrics(df):
            Calculates evaluation metrics based on original and corrected sentences.

    Example Usage:
        ngram_similarity = Levenshtein()
        results_df = ngram_similarity.test("Dataset/Test_Set.csv")
    """

    def __init__(self):
        self.N = 2
        if not nltk.corpus.words.words():
            nltk.download("words")
        self.corpus = set(nltk.corpus.words.words())

    def spellcheck(self, sentence):
        """
        method: spellcheck

        Corrects misspelled words in a sentence.

        args:
            sentence (str): the input sentence to be spellchecked.

        return:
            str: The corrected sentence.

        Example:
            ngram_similarity = Levenshtein()
            corrected_sentence = ngram_similarity.spellcheck("Say hello to my little friend.")
        """

        split = []
        if sentence.endswith("."):
            sentence = sentence[:-1]

        for word in sentence.split():
            closest_word = None

            if word not in self.corpus:
                closest_word = min(
                    self.corpus, key=lambda x: nltk.edit_distance(word, x)
                )

            split.append(closest_word if closest_word else word)

        corrected_sentence = " ".join(split)
        corrected_sentence += "."
        return corrected_sentence

    def test(self, csv_file: str):
        """
        method: test

        Tests the spellcheck method on a dataset and calculates evaluation metrics.

        args:
            csv_file (str): path to the CSV file containing misspelled sentences.

        return:
            df: DataFrame with corrected sentences and evaluation metrics.

        Example:
            results_df = Levenshtein().test("Dataset/Test_Set.csv")
        """
        df = pd.read_csv(csv_file)

        df["Corrected"] = df["Misspelt"].apply(self.spellcheck)

        accuracy, precision, recall_score, f1 = self.calculate_metrics(df)

        print(
            f"{csv_file} - accuracy:{accuracy} , precision:{precision} , recall_score:{recall_score} , f1:{f1}"
        )
        return df

    def calculate_metrics(self, df):
        """
        method: calculate_metrics

        Calculates evaluation metrics based on original and corrected sentences.

        args:
            df (DataFrame): DataFrame containing original and corrected sentences

        return:
            accuracy (float): Accuracy score
            precision (float): Precision score
            recall (float): Recall score
            f1 (float): F1 score

        Example:
            accuracy, precision, recall, f1 = Levenshtein().calculate_metrics(results_df)
        """

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

    def live_correction(self):
        """
        method: live_correction

        Allows live correction of user input using the SpellChecker addon.

        args:
            None

        return:
            None

        Example:
            Norvig().live_correction()
        """

        for x in range(0, 20):
            user_input = input("Input Sentence: ").lower()
            print(self.spellcheck(user_input))


if __name__ == "__main__":
    ngram_similarity = Levenshtein()
    ngram_similarity.test("Dataset/Test_Set.csv")
    # ngram_similarity.live_correction()
