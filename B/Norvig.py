"""
The following code was written as the final project for 
ELEC0141 Deep Learning for Natural Language Processing

Author: Harry R J Softley-Graham
Date: Jan-May 2024

"""

from spellchecker import SpellChecker
import nltk
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk

import pandas as pd


class Norvig:
    """
    class: Norvig

    Implements spellchecking algorithms based on Norvig's approach.

    Attributes:
        N (int): N-gram parameter for spellchecking algorithms.
        corpus (set): Set of words from NLTK's corpus used for spellchecking.

    Methods:
        levenshtein_distance()
        spellcheck()
        spellcheck_addon()
        test()
        live_correction()
        calculate_metrics()

    Example Usage:
        norvig_spellchecker = Norvig()
        norvig_spellchecker.test("Dataset/Test_Set.csv")
    """

    def __init__(self):
        self.N = 1
        if not nltk.corpus.words.words():
            nltk.download("words")
        self.corpus = set(nltk.corpus.words.words())

    def levenshtein_distance(self, s1, s2):
        """
        method: levenshtein_distance

        Calculates the Levenshtein distance between two strings.

        args:
            s1 (str): the first input string.
            s2 (str): the second input string.

        return:
            (int): The Levenshtein distance between s1 and s2.

        Example:
            distance = Norvig().levenshtein_distance("kitten", "sitting")
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
        method: spellcheck

        Corrects misspelled words in a sentence using the Levenshtein distance. Using
        self made algorithm

        args:
            sentence (str): The input sentence to be spellchecked.

        return:
            corrected_sentence (str): prediction for corrected sentence

        Example:
            corrected_sentence = Norvig().spellcheck("Thiss is an exampl of misspeled sentnce.")
        """

        predicts = []
        # Remove the last character if it is a period
        if sentence.endswith("."):
            sentence = sentence[:-1]

        for word in sentence.split():
            suggested_word = word
            min_distance = float("inf")

            for vocab_word in self.corpus:
                distance = self.levenshtein_distance(word, vocab_word)
                if distance < min_distance:
                    min_distance = distance
                    suggested_word = vocab_word

            predicts.append(suggested_word)

        corrected_sentence = " ".join(predicts)
        # Add the period back at the end
        corrected_sentence += "."
        return corrected_sentence

    def spellcheck_addon(self, sentence):
        """
        method: spellcheck_addon

        Corrects misspelled words in a sentence using the SpellChecker module.

        args:
            sentence (str): the input sentence to be spellchecked

        return:
            corrected_sentence (str): prediction for corrected sentence

        Example:
            corrected_sentence = Norvig().spellcheck_addon("Thiss is an exampl of misspeled sentnce.")
        """

        spell = SpellChecker(distance=self.N)
        corrected_words = []

        if sentence.endswith("."):
            sentence = sentence[:-1]

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
        """
        method: test

        tests the spellcheck methods on a dataset and calculates evaluation metrics

        args:
            csv_file (str): path to the CSV file containing misspelled sentences
            addon (bool): if True, use spellcheck_addon method; else, use spellcheck method


        return:
            df (DataFrame): DataFrame with corrected sentences and evaluation metrics

        Example:
            results_df = Norvig().test("./Dataset/Test_Set.csv")
        """

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
            print(self.spellcheck_addon(user_input))

    def calculate_metrics(self, df):
        """
        method: calculate_metrics

        Calculates evaluation metrics based on original and corrected sentences.

        args:
            df (DataFrame): dataFrame containing original and corrected sentences

        return:
            accuracy (float): Accuracy score
            precision (float): Precision score
            recall (float): Recall score
            f1 (float): F1 score

        Example:
            accuracy, precision, recall, f1 = Norvig().calculate_metrics(results_df)
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


if __name__ == "__main__":
    levenshtein = Norvig()
    # levenshtein.test("Dataset/Test_Set.csv")
    levenshtein.live_correction()
