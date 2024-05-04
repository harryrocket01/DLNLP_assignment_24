import subprocess
import sys
import os

# subprocess.check_call([sys.executable, "-m", "pip", "install", ""])


from A.data_synthesizer import DataSynthesizer
from A.tokenise_and_encode import TokeniseAndEncode, Vocabularay

# from B.Levenshtein import *
# from B.Norvig import *
# from B.SeqBasic import *
# from B.SeqAttention import *


from B.Plotting import *

# from A.Model.seq_seq2 import Seq2SeqModel

import pandas as pd
import json


class FinalProject:

    def __init__(self):

        self.script_dir = os.path.dirname(os.path.realpath(__file__))

        self.data_corpus = None

        self.train_out_raw = []
        self.validate_out_raw = []
        self.test_out_raw = []

        self.train_ds = None
        self.val_ds = None
        self.context_text_processor = None
        self.target_text_processor = None
        self.create_data()

        self.graphics()

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

    def lev(self):
        pass

    def norvig(self):
        pass

    def seq_basic(self):
        pass

    def seq_attention(self):
        pass

    def graphics(self):
        self.script_dir
        path_to_save = "/B/Graphics"
        # Final Model
        fig, axs = Plotting().acc_loss_plot(
            root=self.script_dir + "/B/Graphics/ClusterRuns/1714468346_metrics.csv",
            title="Final Model Accuracy-Loss Plot on Full Train Set",
        )

        fig.savefig(self.script_dir + "\B\Graphics\Final_Model_ACCLOSS.pdf")

        # SeqBasic
        fig, axs = Plotting().acc_loss_plot(
            root=self.script_dir + "/B/Graphics/ClusterRuns/1714209146_metrics.csv",
            title="Seq2Seq (no attention) Accuracy-Loss Plot",
        )

        fig.savefig(self.script_dir + "\B\Graphics\SeqBasic_ACCLOSS.pdf")
        # cell plots
        fig1, axs1 = Plotting().acc_loss_plot(
            root=self.script_dir + "/B/Graphics/ClusterRuns/1713795920_metrics.csv",
        )
        fig2, axs2 = Plotting().acc_loss_plot(
            root=self.script_dir + "/B/Graphics/ClusterRuns/1713803259_metrics.csv",
        )
        fig3, ax3 = Plotting().acc_loss_plot(
            root=self.script_dir + "/B/Graphics/ClusterRuns/1713818918_metrics.csv",
        )
        fig1.savefig(self.script_dir + "\B\Graphics\RNN_ACCLOSS.pdf")
        fig2.savefig(self.script_dir + "\B\Graphics\GRU_ACCLOSS.pdf")
        fig3.savefig(self.script_dir + "\B\Graphics\LSTM_ACCLOSS.pdf")

        fig, axs = Plotting().decoder_cell_plot(
            root1=self.script_dir + "/B/Graphics/ClusterRuns/1713795920_metrics.csv",
            root2=self.script_dir + "/B/Graphics/ClusterRuns/1713803259_metrics.csv",
            root3=self.script_dir + "/B/Graphics/ClusterRuns/1713818918_metrics.csv",
        )
        fig.savefig(self.script_dir + "/B/Graphics/decoder_cell_comparison.pdf")

        # attention
        fig1, axs1 = Plotting().acc_loss_plot(
            root=self.script_dir + "/B/Graphics/ClusterRuns/1713795920_metrics.csv",
        )
        fig2, axs2 = Plotting().acc_loss_plot(
            root=self.script_dir + "/B/Graphics/ClusterRuns/1713803259_metrics.csv",
        )
        fig1.savefig(self.script_dir + "/B/Graphics//bahdanau_ACCLOSS.pdf")
        fig2.savefig(self.script_dir + "/B/Graphics//loung_ACCLOSS.pdf")

        fig, axs = Plotting().attention_plot(
            root1=self.script_dir + "/B/Graphics/ClusterRuns/1713821129_metrics.csv",
            root2=self.script_dir + "/B/Graphics/ClusterRuns/1713825080_metrics.csv",
        )

        fig.savefig(self.script_dir + "/B/Graphics//attention_comparison.pdf")


if __name__ == "__main__":
    FinalProject()
