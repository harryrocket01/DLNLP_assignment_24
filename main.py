import subprocess
import sys
import os

# subprocess.check_call([sys.executable, "-m", "pip", "install", ""])


from A.data_synthesizer import DataSynthesizer
from A.tokenise_and_encode import TokeniseAndEncode, Vocabularay

# from A.Model.seq_seq2 import Seq2SeqModel

import pandas as pd
import json


class FinalProject:

    def __init__(self):

        self.data_corpus = None

        self.train_out_raw = []
        self.validate_out_raw = []
        self.test_out_raw = []

        self.train_ds = None
        self.val_ds = None
        self.context_text_processor = None
        self.target_text_processor = None
        self.create_data()
        # self.process_tv_set()
        # self.define_model()

    def create_data(self, download=False):
        file_name = "Misspelling_Corpus.csv"
        root = ".\Dataset\Sentences\GenericsKB-Best.tsv"
        mistake_root = [
            ".\Dataset\\Mistakes\\en_keystroke_pairs.sorted.txt",
            ".\Dataset\\Mistakes\\aspell.dat",
            ".\Dataset\\Mistakes\\missp.dat",
            ".\Dataset\\Mistakes\\wikipedia.dat",
        ]

        if os.path.isfile(f"./Dataset/{file_name}") == True:
            print("Dataset Already Created at root:")
            print(f"./Dataset/{file_name}")
            return 1
        elif download == True:
            DataSynthesizer().download_files()

        data_create = DataSynthesizer()

        root_flag = data_create.set_root(mistake_root)

        if root_flag is not None:
            if download == True:
                DataSynthesizer().download_files()
            else:
                data_create.read_all()
                data_create.create_misspell_corpus(
                    root,
                    unique_sentences=10000,
                    sentence_variation=1,
                    file_name=file_name,
                )

        else:
            return 0
        return 1

    def process_tv_set(self):
        self.data_corpus = pd.read_csv(
            f"./Dataset/Misspelling_Corpus.csv", nrows=100000
        )

        origin_test = self.data_corpus["Original"].to_numpy()
        target_test = self.data_corpus["Misspelt"].to_numpy()
        vocab = Vocabularay().generate_vocab(origin_test)
        print(vocab)

        (
            self.train_ds,
            self.val_ds,
            self.context_text_processor,
            self.target_text_processor,
        ) = TokeniseAndEncode(vocab).convert_list_to_tensor(
            origin_test, target_test, buffer=100000
        )

    def Lev():
        pass

    def Norvig():
        pass

    def create_metrics():
        pass


if __name__ == "__main__":
    FinalProject()
