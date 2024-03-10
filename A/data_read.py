import numpy as np
import re
import random
import pandas as pd


class DataRead:

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


class FullDataRead:
    def __init__(self):

        self.root_files = [
            ".\Dataset\en_keystroke_pairs.sorted.txt",
            ".\Dataset\\aspell.dat",
            ".\Dataset\\missp.dat",
            ".\Dataset\\wikipedia.dat",
        ]
        self.mispell_dict = {}

    def read_all(self):

        # read files merge dictionaries
        for current in self.root_files:

            if ".txt" in current:
                current_dic = DataRead().read_txt(current)
            else:
                current_dic = DataRead().read_dat(current)

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

    def read_sentances(self):
        root = ".\Dataset\Word\GenericsKB-Best.tsv"
        sentence_dataframe = DataRead().read_tsv(root)
        sentence_dataframe["MISSPELT SENTENCE"] = sentence_dataframe["GENERIC SENTENCE"]
        sentence_dataframe["MISSPELT SENTENCE"] = sentence_dataframe[
            "MISSPELT SENTENCE"
        ].apply(self.mispell)

        print(sentence_dataframe)

    def mispell(self, sentence):

        sentence_list = re.findall(r"\w+|\S", sentence)

        random_index = random.randint(0, len(sentence_list) - 1)
        while random_index < len(sentence_list):
            word = sentence_list[random_index]
            if word in self.mispell_dict:
                random_value = random.choice(self.mispell_dict[word])
                sentence_list[random_index] = random_value
                break
            random_index += 1
        merged_sentence = " ".join(sentence_list)
        merged_sentence = re.sub(r"\s+([.,!?])", r"\1", merged_sentence)
        return merged_sentence


test_inst = FullDataRead()
test = test_inst.read_all()
test2 = test_inst.read_sentances()
