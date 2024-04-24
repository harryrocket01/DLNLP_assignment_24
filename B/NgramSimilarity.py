import sys
import subprocess


subprocess.check_call([sys.executable, "-m", "pip", "install", "ngram"])

from ngram import NGram
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import nltk

nltk.download("words")


class NgramSimilarity:
    def __init__(self):
        self.N = 2
        if not nltk.corpus.words.words():
            nltk.download("words")
        self.corpus = set(nltk.corpus.words.words())

    def spellcheck(self, sentence):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """

        split = []

        for word in sentence.split():
            closest_word = None

            if word not in self.corpus:
                closest_word = min(
                    self.corpus, key=lambda x: nltk.edit_distance(word, x)
                )

            split.append(closest_word if closest_word else word)

        prediction = " ".join(split)
        prediction = prediction + "."
        return prediction

    def test(self, csv_file):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """
        print("1")
        df = pd.read_csv(csv_file)

        print("1")

        df["Corrected"] = df["Misspelt"].apply(self.spellcheck)
        print("1")

        accuracy, precision, recall_score, f1 = self.calculate_metrics(df)
        print("1")

        print(
            f"{csv_file} - accuracy:{accuracy} , precision:{precision} , recall_score:{recall_score} , f1:{f1}"
        )
        return df

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
    ngram_similarity = NgramSimilarity()
    ngram_similarity.test("Dataset/Test_Set.csv")
