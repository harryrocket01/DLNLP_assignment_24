import numpy as np
import re
import random
import pandas as pd
import os
import requests
import shutil
import zipfile


class FileRead:

    def __init__(self):
        pass

    def read_txt(self, root: str):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """

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
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """

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

    def read_tsv(self, root, rows=1000000):
        df = pd.read_csv(root, sep="\t", nrows=rows)
        return df


class DataSynthesizer:
    def __init__(self):

        self.root_files = []

        self.mistake_freq = {0: 0, 1: 10, 2: 4, 3: 2, 4: 1, 5: 1}

        self.mispell_dict = {}

    def download_files(self):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """

        file_id = "1bK6PvtUpluk-4WIpEhs7p2c3vy5AVwVG"
        url_dataset = f"https://drive.usercontent.google.com/download?id=1bK6PvtUpluk-4WIpEhs7p2c3vy5AVwVG&export=download&authuser=0&confirm=t&uuid=0050d7b2-f732-42fd-a356-fafc4aa33e4f&at=APZUnTW3LQrDM25YzKQbJIzb42qL%3A1712155394334"

        self.http_fetch_and_unzip(url_dataset, file_id)

    def http_fetch_and_unzip(self, url, filename):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """

        print("Fetching Zip")
        zip_filename = f"{filename}.zip"

        try:
            with requests.get(url, stream=True) as r:
                with open(zip_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            print(f"Error downloading file: {e}")
            return

        with open(zip_filename, "rb") as f:
            file_header = f.read(1024)
        if b"DOCTYPE html" in file_header:
            print("Downloaded file is the warning page HTML, not the actual file.")
            os.remove(zip_filename)  # Remove the HTML file
            return

        print("Extracting Zip")
        extract_to = "./Dataset"
        try:
            if not os.path.exists(extract_to):
                os.makedirs(extract_to)

            with zipfile.ZipFile(zip_filename, "r") as zip_ref:
                zip_ref.extractall(extract_to)
        except Exception as e:
            print(f"Error extracting zip file: {e}")
            return

        print("Cleaning up files")
        # Remove the zip file
        try:
            os.remove(zip_filename)
        except Exception as e:
            print(f"Error removing zip file: {e}")
            return

        print("Completed")

    def set_root(self, root):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """
        for path in root:
            if os.path.isfile(path) == False:
                print("Missing File At Root")
                print(path)
                return None
        self.root_files = root
        return 1

    def read_all(self):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """
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

        return self.mispell_dict

    def create_misspell_corpus(
        self,
        root,
        unique_sentences=2000,
        sentence_variation=50,
        file_name="Misspelling_Corpus.csv",
    ):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """
        if os.path.isfile(root) == False:
            print("Missing File at Root")
            print({root})
            return None

        sentence_dataframe = FileRead().read_tsv(root)
        # shuffle and sort
        sentence_dataframe = (
            sentence_dataframe.sample(frac=1, random_state=76)  # Shuffle the DataFrame
            .reset_index(drop=True)  # Reset the index
            .sort_values(by="SCORE", ascending=False)  # Sort by the "SCORE" column
            .reset_index(drop=True)  # Reset the index again
        )

        sentence_dataframe = sentence_dataframe.head(unique_sentences)

        print(sentence_dataframe)
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

        sentence_dataframe = sentence_dataframe.loc[
            sentence_dataframe.index.repeat(sentence_variation)
        ].reset_index(drop=True)

        sentence_dataframe["Misspelt"] = sentence_dataframe["Misspelt"].apply(
            self.mispell
        )

        sentence_dataframe["Flag"] = sentence_dataframe.apply(
            lambda row: self.flag(row["Original"], row["Misspelt"]), axis=1
        )

        sentence_dataframe.to_csv(f"./Dataset/{file_name}")

    def format_to_vocab(self, sentence):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """
        sentence = sentence.lower()
        sentence = self.clean_text(sentence)
        sentence_list = re.findall(r"\w+|\S", sentence)
        merged_sentence = " ".join(sentence_list)
        merged_sentence = re.sub(r"\s+([.,!?])", r"\1", merged_sentence)
        return merged_sentence

    def clean_text(self, text):
        """Remove unwanted characters and extra spaces from the text"""
        text = re.sub(r"([?.!,¿])", r" \1 ", text)
        text = re.sub(r'[" "]+', " ", text)
        text = re.sub(r"\n", " ", text)
        text = re.sub(r"[{}@\"_*>()\\#%+=\[\]]", "", text)
        text = re.sub("'", "", text)
        text = re.sub(" +", " ", text)  # Removes extra spaces
        return text

    def mispell(self, sentence):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """

        sample = self.sample_from_dict()

        merged_sentence = sentence

        if sample == 0:
            sentence_list = re.findall(r"\w+|\S", sentence)
            merged_sentence = " ".join(sentence_list)
            merged_sentence = re.sub(r"\s+([.,!?])", r"\1", merged_sentence)
            return merged_sentence

        for x in range(sample):

            sentence_list = re.findall(r"\w+|\S", merged_sentence)
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

    def sample_from_dict(self):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """
        total_weight = sum(self.mistake_freq.values())
        rand_num = random.uniform(0, total_weight)
        cumulative_weight = 0
        for key, weight in self.mistake_freq.items():
            cumulative_weight += weight
            if rand_num < cumulative_weight:
                return key

    def flag(self, original, misspelt):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """

        return 1 if original != misspelt else 0
