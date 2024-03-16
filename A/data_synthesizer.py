import numpy as np
import re
import random
import pandas as pd


class FileRead:

    def __init__(self):
        pass

    def read_txt(self, root: str):

        misspelling_dict = {}
        with open(root, "r") as file:
            for line in file:
                incorrect, correct = line.strip().split("\t")

                incorrect = re.sub(r"[^a-zA-Z]", "", incorrect)
                correct = re.sub(r"[^a-zA-Z]", "", correct)

                if correct in misspelling_dict:
                    misspelling_dict[correct].append(incorrect)
                else:
                    misspelling_dict[correct] = [incorrect]

        return misspelling_dict

    def read_dat(self, root: str):
        data_dict = {}
        with open(root, "r") as file:
            for line in file:
                if line.startswith("$"):
                    correct_spelling = line.strip()[1:]
                    data_dict[correct_spelling] = []
                else:
                    misspelling = line.strip()
                    data_dict[correct_spelling].append(misspelling)

        return data_dict

    def read_tsv(self, root):
        df = pd.read_csv(root, sep="\t")
        return df


class DataSynthesizer:
    def __init__(self):

        self.root_files = [
            ".\Dataset\\Mistakes\\en_keystroke_pairs.sorted.txt",
            ".\Dataset\\Mistakes\\aspell.dat",
            ".\Dataset\\Mistakes\\missp.dat",
            ".\Dataset\\Mistakes\\wikipedia.dat",
        ]

        self.mistakes = 10

        self.mispell_dict = {}

    def read_all(self):

        # read files merge dictionaries
        for current in self.root_files:

            if ".txt" in current:
                current_dic = FileRead().read_txt(current)
            else:
                current_dic = FileRead().read_dat(current)

            for key, value in current_dic.items():
                if key in self.mispell_dict:
                    self.mispell_dict[key].extend(value)
                else:
                    self.mispell_dict[key] = value

        # remove duplicate
        self.mispell_dict = {
            key: self.mispell_dict[key] for key in sorted(self.mispell_dict)
        }

        for key, value in self.mispell_dict.items():

            self.mispell_dict[key] = list(dict.fromkeys(value))

        print(self.mispell_dict)

        return self.mispell_dict

    def read_sentences(self):
        root = ".\Dataset\Sentences\GenericsKB-Best.tsv"
        sentence_dataframe = FileRead().read_tsv(root)
        sentence_dataframe = sentence_dataframe.drop(
            ["SOURCE", "TERM", "SCORE", "QUANTIFIER"], axis=1
        )
        sentence_dataframe = sentence_dataframe.rename(
            {"GENERIC SENTENCE": "Original"}, axis="columns"
        )
        sentence_dataframe["Original"] = sentence_dataframe["Original"].apply(
            self.format_to_vocab
        )
        sentence_dataframe["Misspelt"] = sentence_dataframe["Original"]

        for x in range(0, self.mistakes):
            sentence_dataframe["Misspelt"] = sentence_dataframe["Misspelt"].apply(
                self.mispell
            )

        sentence_dataframe["Flag"] = sentence_dataframe.apply(
            lambda row: self.flag(row["Original"], row["Misspelt"]), axis=1
        )

        print(sentence_dataframe)
        sentence_dataframe.to_csv("./Dataset/Misspelling_Corpus.csv")

    def format_to_vocab(self, sentence):
        sentence = sentence.lower()
        sentence = self.clean_text(sentence)
        sentence_list = re.findall(r"\w+|\S", sentence)
        merged_sentence = " ".join(sentence_list)
        merged_sentence = re.sub(r"\s+([.,!?])", r"\1", merged_sentence)
        return merged_sentence

    def clean_text(self, text):
        """Remove unwanted characters and extra spaces from the text"""
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"[{}@\"_*>()\\#%+=\[\]]", "", text)
        text = re.sub("'", "", text)
        text = re.sub(" +", " ", text)  # Removes extra spaces
        return text

    def mispell(self, sentence):

        # Probability of a mistake appearing within the dataset
        if random.randint(1, self.mistakes) != 1:
            return sentence

        sentence_list = re.findall(r"\w+|\S", sentence)
        shuffle_list = np.array(sentence_list)
        random.shuffle(shuffle_list)

        for word in shuffle_list:
            if word in self.mispell_dict:
                mispell_choice = random.choice(self.mispell_dict[word])
                sentence_list[sentence_list.index(word)] = mispell_choice
                break
        merged_sentence = " ".join(sentence_list)
        merged_sentence = re.sub(r"\s+([.,!?])", r"\1", merged_sentence)
        return merged_sentence

    def flag(self, original, misspelt):
        return 1 if original != misspelt else 0


if __name__ == "__main__":
    test_inst = DataSynthesizer()
    test = test_inst.read_all()
    test2 = test_inst.read_sentences()
