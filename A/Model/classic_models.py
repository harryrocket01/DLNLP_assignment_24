import subprocess
import sys
import os

subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])


from ngram import NGram
import re
import os
import string

from spellchecker import SpellChecker

import nltk

nltk.download("words")


class BenchMark:
    def __init__(self):
        self.N = 2
        if not nltk.corpus.words.words():
            nltk.download("words")
        self.corpus = set(nltk.corpus.words.words())

    def check_ngrams(self, sentences):
        predicts = []

        for sentence in sentences:
            split = []

            for word in sentence.split():
                if word not in self.corpus:

                    closest_word = min(
                        self.corpus, key=lambda x: nltk.edit_distance(word, x)
                    )

                split.append(closest_word if closest_word else word)

            predicts.append(" ".join(split))

        return predicts

    def check_nor(self, sentences):
        """
        It uses a Levenshtein Distance algorithmthe frequency list are more likely the correct results.
        """
        spell = SpellChecker(distance=self.N)
        predicts = []

        for sentence in sentences:
            split = []

            for word in sentence.split():

                misspelled = spell.correction(word)
                split.append(misspelled)

            predicts.append(" ".join(split))

        return predicts


# Example usage
spell_checker = BenchMark()

sentences = [
    "Ths is a smaple setence for speel checking.",
    "Anothe sentance wit speling mistaks.",
]

corrected_sentences = spell_checker.check_nor(sentences)
print(corrected_sentences)
