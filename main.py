"""
The following code was written as the final project for 
ELEC0141 Deep Learning for Natural Language Processing

This is the main function, used to run and access various
models used within this project.

Author: Harry R J Softley-Graham
Date: Jan-May 2024

"""

import subprocess
import sys

import os
import warnings

warnings.filterwarnings("ignore")


from A.data_synthesizer import DataSynthesizer

from B.Levenshtein import *
from B.Norvig import *
from B.SeqBasic import *
from B.SeqAttention import *
from B.Plotting import *


class FinalProject:
    """
    class: FinalProject

    Main class that call all the models built and used within,
    this project.

    args:
        model(str): what model should be run
        new_dataset(bool): flag for if the dataset should be remade
    methods:
        create_data(): creates and download dataset
        lev(): runs levenshtein basic algorithm
        norvig(): runs norvig algorithm
        seq_basic(): runs seq_basic algorithm
        seq_attention(): runs seq attention algorithm
        final_model(): runs the final model for the project
        graphics(): creates graphs and graphics

    Example:
        FinalProject()
    """

    def __init__(self, model: str = None, new_dataset: bool = False):

        self.script_dir = os.path.dirname(os.path.realpath(__file__))

        self.data_corpus = None

        self.train_out_raw = []
        self.validate_out_raw = []
        self.test_out_raw = []

        self.train_ds = None
        self.val_ds = None
        self.context_text_processor = None
        self.target_text_processor = None
        self.create_data(overwrite=new_dataset)

        self.graphics()
        try:
            model = model.lower()
        except:
            pass

        if model == "levenshtein":
            self.lev()
        elif model == "norvig":
            self.norvig()
        elif model == "seqbasic":
            self.seq_basic()
        elif model == "seqattention":
            self.seq_attention()
        elif model == "final":
            self.final_model()
        else:
            self.final_model()

    def create_data(self, download: bool = True, overwrite: bool = False):
        """
        method:create_data

        Creates and sets up dataset through calling data synthesizer tool.

        args:
            download (bool): flag if it should download database
            overwrite (bool): flag if data should be overwitten with new dataset
        return:
            (bool): flag on functions success
        Example:
            FinalProject.create_data()
        """

        file_name = "Misspelling_Corpus.csv"
        root = ".\Dataset\Sentences\GenericsKB-Best.tsv"

        unique_sentences = 10, 000
        sentence_variation = 50
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

        if (root_flag is not None) or (overwrite == True):
            if download == True:
                DataSynthesizer().download_files()
            else:
                data_create.read_all()
                data_create.create_misspell_corpus(
                    root,
                    unique_sentences=unique_sentences,
                    sentence_variation=sentence_variation,
                    file_name=file_name,
                )

        else:
            return 0
        return 1

    def lev(self):
        """
        method:lev

        Runs the Levenshtein model and tests it.

        args:
            None
        return:
            None
        Example:
            FinalProject().lev()

        """
        levenshtein = Levenshtein()
        levenshtein.test("Dataset/Test_Set.csv")

    def norvig(self):
        """
        method:norvig

        Runs the norvig model and tests it.

        args:
            None
        return:
            None
        Example:
            FinalProject().norvig()
        """
        norvig = Norvig()
        norvig.test("Dataset/Test_Set.csv")

    def seq_basic(self):
        """
        method:seq_basic

        Runs the basic sequence 2 sequnce model and runs it.

        args:
            None
        return:
            None
        Example:
            FinalProject().seq_basic()
        """
        buffer = 131072
        batch_size = 64
        num_examples = 100, 000
        learning_rate = 0.0001
        file_dir = "./B/training"
        train_dir = "Misspelling_Corpus.csv"
        test_dir = "Test_Set.csv"

        inst = SeqBasic(
            buffer=buffer,
            batch_size=batch_size,
            num_examples=num_examples,
            learning_rate=learning_rate,
            file_dir=file_dir,
            train_dir=train_dir,
            test_dir=test_dir,
        )
        inst.train(epochs=10)

    def seq_attention(self):
        """
        method:seq_attention

        Runs the Sequence 2 Sequence model with attention block and tests model.

        args:
            None
        return:
            None
        Example:
            FinalProject().seq_attention()
        """
        buffer = 131072
        batch_size = 64
        num_examples = 100, 000
        learning_rate = 0.0001
        file_dir = "./B/training"
        attention_type = "luong"
        encoder_cell = "LSTM"
        decoder_cell = "LSTM"
        train_dir = "Misspelling_Corpus.csv"
        test_dir = "Test_Set.csv"

        inst = SeqAttention(
            buffer=buffer,
            batch_size=batch_size,
            num_examples=num_examples,
            learning_rate=learning_rate,
            file_dir=file_dir,
            attention_type=attention_type,
            encoder_cell=encoder_cell,
            decoder_cell=decoder_cell,
            train_dir=train_dir,
            test_dir=test_dir,
        )
        inst.train(epochs=10)

    def final_model(self):
        """
        method:final_model

        Runs the Levenshtein model.

        args:
            None
        return:
            None
        Example:
            FinalProject().final_model()
        """
        buffer = 131072
        batch_size = 64
        num_examples = 500000
        learning_rate = 0.0001
        file_dir = "./B/training"
        attention_type = "luong"
        encoder_cell = "LSTM"
        decoder_cell = "GRU"
        train_dir = "Misspelling_Corpus.csv"
        test_dir = "Test_Set.csv"

        inst = SeqAttention(
            buffer=buffer,
            batch_size=batch_size,
            num_examples=num_examples,
            learning_rate=learning_rate,
            file_dir=file_dir,
            attention_type=attention_type,
            encoder_cell=encoder_cell,
            decoder_cell=decoder_cell,
            train_dir=train_dir,
            test_dir=test_dir,
        )
        inst.train(epochs=20)
        inst.test()

    def graphics(self):
        """
        method:graphics

        Creates all of the plots from the given run data.

        args:
            none
        return:
            none
        Example:
            FinalProject().graphics()

        """
        self.script_dir
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
    if len(sys.argv) == 1:
        FinalProject(model=None, new_dataset=False)

    elif len(sys.argv) == 3:
        model_name = sys.argv[1]
        new_dataset = sys.argv[2]
        if (not isinstance(model_name, str)) or (not isinstance(new_dataset, bool)):
            FinalProject(model=model_name, new_dataset=new_dataset)

    else:
        print(
            "Usage: python main.py <model_name> [new_dataset]\nOR\nUSAGE: python main.py"
        )
