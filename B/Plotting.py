import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.typing import ArrayLike
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Plotting:

    def __init__(self) -> None:
        pass

    def Confusion_Matrix(
        self, true_labels: ArrayLike, pred_labels: ArrayLike, title: str = ""
    ) -> None:
        """
        function: confusion_matrix

        Creates a confusion matrix plot from true and predicted labels

        args:
            true_labels (ArrayLike): Array of True Labels
            pred_labels (ArrayLike): Array of Predicted Labels
            title (string): Title for plot

        return:
            fig (plt.Figure): Matplotlib Figure Class of plot
            axs (plt.Axes): Matplotlib Axes Class of plot

        Example:
            fig, axs = Plotting(Dataset, Labels).confusion_matrix(true_labels,pred_labels,title)
        """
        cm = confusion_matrix(true_labels, pred_labels)

        fig, axs = plt.subplots(figsize=(4, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="crest",
            cbar=False,
            xticklabels=np.unique(true_labels),
            yticklabels=np.unique(true_labels),
        )
        fig.set_tight_layout(True)
        axs.set_title(title)

        axs.set_xlabel("Actual Values")
        axs.set_ylabel("Predicted Values")

        return fig, axs

    def Line_Plot(
        self, x: ArrayLike, y: ArrayLike, title: str, x_label: str, y_label: str, legend
    ):
        """
        function: line_plot

        Creates a line plot from two inputted arrays

        args:
            x (ArrayLike): array to plot on x axis
            y (ArrayLike): array to plot on y axis
            title (str): title for plot
            x_label (str): X axis Label
            y_label (str): Y axis Label
            legend (ArrayLike): Array for plot legend

        return:
            fig (plt.Figure): Matplotlib Figure Class of plot
            axs (plt.Axes): Matplotlib Axes Class of plot

        Example:
            fig, axs = Plotting(Dataset, Labels).line_plot(x,y,title)
        """
        fig, axs = plt.subplots()
        counter = 1

        if np.array(y).ndim >= 2:
            print(True)
            for Y_current in y:
                axs.plot(x, Y_current)
        else:
            axs.plot(x, y)

        axs.set_title(title)
        axs.set_xlabel(x_label)
        axs.set_ylabel(y_label)
        axs.legend(legend)
        fig.set_tight_layout(True)
        fig.set_size_inches(3.5, 2.5)

        return fig, axs

    def acc_loss_plot(self, root: str, title="Accuracy Loss Plot"):
        """
        function: acc_loss_plot

        Creates a two y-axis accuracy-loss plot from inputted arrays. Used
        to show training of a model

        args:
            acc (ArrayLike): Array of train accuracy points
            loss (ArrayLike): Array of train loss points
            val_acc (ArrayLike): Array of validation accuracy points
            val_loss (ArrayLike): Array of validation loss points
            x_axis (ArrayLike): array of epochs
            title (str): Title for plot

        return:
            fig (plt.Figure): Matplotlib Figure Class of plot
            axs (plt.Axes): Matplotlib Axes Class of plot

        Example:
            fig, axs = Plotting(Dataset, Labels).acc_loss_plot(acc,loss,v_acc,v_loss)
        """

        dataframe = pd.read_csv(root)

        # losses	accuracies	val_losses	val_accuracies

        dataframe["losses"]
        fig, axs = plt.subplots()
        axs2 = axs.twinx()
        x_axis = np.linspace(
            1, len(dataframe["accuracies"]), len(dataframe["accuracies"]), dtype=int
        )

        axs.set_xlabel("Epochs")
        axs.set_ylabel("Accuracy", color="tab:red")
        axs.plot(
            x_axis, dataframe["accuracies"], color="tab:red", label="Train Accuracy"
        )
        axs.plot(
            x_axis,
            dataframe["val_accuracies"],
            color="tab:red",
            alpha=0.6,
            label="Val Accuracy",
        )

        axs.tick_params(axis="y", labelcolor="tab:red")

        axs2.set_ylabel("Loss", color="tab:blue")
        axs2.plot(x_axis, dataframe["losses"], color="tab:blue", label="Train Loss")
        axs2.plot(
            x_axis,
            dataframe["val_losses"],
            color="tab:blue",
            alpha=0.6,
            label="Val Loss",
        )
        axs2.tick_params(axis="y", labelcolor="tab:blue")

        lines, labels = axs.get_legend_handles_labels()
        lines2, labels2 = axs2.get_legend_handles_labels()
        axs2.legend(lines + lines2, labels + labels2, loc="lower right")

        axs.set_title(title)

        fig.set_size_inches(5.5, 3)
        fig.set_tight_layout(True)

        return fig, axs

    def acc_loss_log_plot(self, root: str, title="Accuracy Loss Plot"):
        """
        function: acc_loss_plot

        Creates a two y-axis accuracy-loss plot from inputted arrays. Used
        to show training of a model

        args:
            acc (ArrayLike): Array of train accuracy points
            loss (ArrayLike): Array of train loss points
            val_acc (ArrayLike): Array of validation accuracy points
            val_loss (ArrayLike): Array of validation loss points
            x_axis (ArrayLike): array of epochs
            title (str): Title for plot

        return:
            fig (plt.Figure): Matplotlib Figure Class of plot
            axs (plt.Axes): Matplotlib Axes Class of plot

        Example:
            fig, axs = Plotting(Dataset, Labels).acc_loss_plot(acc,loss,v_acc,v_loss)
        """

        dataframe = pd.read_csv(root)

        fig, axs = plt.subplots()
        axs2 = axs.twinx()
        x_axis = np.linspace(
            1, len(dataframe["accuracies"]), len(dataframe["accuracies"]), dtype=int
        )

        axs.set_xlabel("Epochs")
        axs.set_ylabel("Accuracy", color="tab:red")
        axs.plot(
            x_axis, dataframe["accuracies"], color="tab:red", label="Train Accuracy"
        )
        axs.plot(
            x_axis,
            dataframe["val_accuracies"],
            color="tab:red",
            alpha=0.6,
            label="Val Accuracy",
        )

        axs.tick_params(axis="y", labelcolor="tab:red")
        axs.set_yscale("log")  # Setting y-axis to log scale for accuracy

        axs2.set_ylabel("Loss", color="tab:blue")
        axs2.plot(x_axis, dataframe["losses"], color="tab:blue", label="Train Loss")
        axs2.plot(
            x_axis,
            dataframe["val_losses"],
            color="tab:blue",
            alpha=0.6,
            label="Val Loss",
        )
        axs2.tick_params(axis="y", labelcolor="tab:blue")
        axs2.set_yscale("log")  # Setting y-axis to log scale for loss

        lines, labels = axs.get_legend_handles_labels()
        lines2, labels2 = axs2.get_legend_handles_labels()
        axs2.legend(lines + lines2, labels + labels2, loc="lower right")

        axs.set_title(title)

        fig.set_size_inches(5.5, 3)
        fig.set_tight_layout(True)

        return fig, axs

    def acc_loss_log_plot(self, root: str, title="Accuracy Loss Plot"):
        """
        function: acc_loss_plot

        Creates a two y-axis accuracy-loss plot from inputted arrays. Used
        to show training of a model

        args:
            acc (ArrayLike): Array of train accuracy points
            loss (ArrayLike): Array of train loss points
            val_acc (ArrayLike): Array of validation accuracy points
            val_loss (ArrayLike): Array of validation loss points
            x_axis (ArrayLike): array of epochs
            title (str): Title for plot

        return:
            fig (plt.Figure): Matplotlib Figure Class of plot
            axs (plt.Axes): Matplotlib Axes Class of plot

        Example:
            fig, axs = Plotting(Dataset, Labels).acc_loss_plot(acc,loss,v_acc,v_loss)
        """

        dataframe = pd.read_csv(root)

        fig, axs = plt.subplots()
        axs2 = axs.twinx()
        x_axis = np.linspace(
            1, len(dataframe["accuracies"]), len(dataframe["accuracies"]), dtype=int
        )

        axs.set_xlabel("Epochs")
        axs.set_ylabel("Accuracy", color="tab:red")
        axs.plot(
            x_axis, dataframe["accuracies"], color="tab:red", label="Train Accuracy"
        )
        axs.plot(
            x_axis,
            dataframe["val_accuracies"],
            color="tab:red",
            alpha=0.6,
            label="Val Accuracy",
        )

        axs.tick_params(axis="y", labelcolor="tab:red")
        axs.set_yscale("log")  # Setting y-axis to log scale for accuracy

        axs2.set_ylabel("Loss", color="tab:blue")
        axs2.plot(x_axis, dataframe["losses"], color="tab:blue", label="Train Loss")
        axs2.plot(
            x_axis,
            dataframe["val_losses"],
            color="tab:blue",
            alpha=0.6,
            label="Val Loss",
        )
        axs2.tick_params(axis="y", labelcolor="tab:blue")
        axs2.set_yscale("log")  # Setting y-axis to log scale for loss

        lines, labels = axs.get_legend_handles_labels()
        lines2, labels2 = axs2.get_legend_handles_labels()
        axs2.legend(lines + lines2, labels + labels2, loc="lower right")

        axs.set_title(title)

        fig.set_size_inches(5.5, 3)
        fig.set_tight_layout(True)

        return fig, axs

    def decoder_cell_plot(
        self, root1: str, root2: str, root3: str, title="Decoder Cell Comparison"
    ):
        """
        function: acc_loss_plot


        args:


        return:
            fig (plt.Figure): Matplotlib Figure Class of plot
            axs (plt.Axes): Matplotlib Axes Class of plot

        Example:
            fig, axs = Plotting(Dataset, Labels).acc_loss_plot(acc,loss,v_acc,v_loss)
        """

        dataframe1 = pd.read_csv(root1)
        dataframe2 = pd.read_csv(root2)
        dataframe3 = pd.read_csv(root3)

        fig, axs = plt.subplots()
        axs2 = axs.twinx()
        x_axis = np.linspace(
            1, len(dataframe1["accuracies"]), len(dataframe1["accuracies"]), dtype=int
        )

        axs.set_xlabel("Epochs")
        axs.set_ylabel("Train Accuracy", color="tab:red")
        axs.plot(x_axis, dataframe1["accuracies"], color="tab:red", label="RNN Acc")
        axs.plot(
            x_axis,
            dataframe2["accuracies"],
            color="tab:red",
            alpha=0.65,
            label="GRU Acc",
        )
        axs.plot(
            x_axis,
            dataframe3["accuracies"],
            color="tab:red",
            alpha=0.35,
            label="LSTM Acc",
        )

        axs.tick_params(axis="y", labelcolor="tab:red")

        axs2.set_ylabel("Loss", color="tab:blue")
        axs2.plot(
            x_axis, dataframe1["losses"], color="tab:blue", label="RNN Loss", alpha=0.6
        )
        axs2.plot(
            x_axis,
            dataframe2["losses"],
            color="tab:blue",
            alpha=0.65,
            label="GRU Loss",
        )
        axs2.plot(
            x_axis,
            dataframe3["losses"],
            color="tab:blue",
            alpha=0.35,
            label="LSTM Loss",
        )
        axs2.tick_params(axis="y", labelcolor="tab:blue")

        lines, labels = axs.get_legend_handles_labels()
        lines2, labels2 = axs2.get_legend_handles_labels()
        axs2.legend(lines + lines2, labels + labels2, loc="lower right")

        axs.set_title(title)

        fig.set_size_inches(5.5, 3)
        fig.set_tight_layout(True)

        return fig, axs

    def attention_plot(
        self, root1: str, root2: str, title="Attention Block Comparison"
    ):
        """
        function:


        args:
        return:
            fig (plt.Figure): Matplotlib Figure Class of plot
            axs (plt.Axes): Matplotlib Axes Class of plot

        Example:
            fig, axs = Plotting(Dataset, Labels).acc_loss_plot(acc,loss,v_acc,v_loss)
        """

        dataframe1 = pd.read_csv(root1)
        dataframe2 = pd.read_csv(root2)

        fig, axs = plt.subplots()
        axs2 = axs.twinx()
        x_axis = np.linspace(
            1, len(dataframe1["accuracies"]), len(dataframe1["accuracies"]), dtype=int
        )

        axs.set_xlabel("Epochs")
        axs.set_ylabel("Train Accuracy", color="tab:red")
        axs.plot(x_axis, dataframe1["accuracies"], color="tab:red", label="Bahdanau ")
        axs.plot(
            x_axis,
            dataframe2["accuracies"],
            color="tab:red",
            alpha=0.6,
            label="Luong",
        )

        axs.tick_params(axis="y", labelcolor="tab:red")

        axs2.set_ylabel("Time Elapsed Per Epoch (s)", color="tab:blue")
        axs2.plot(x_axis, dataframe1["time"], color="tab:blue", label="Bahdanau ")
        axs2.plot(
            x_axis,
            dataframe2["time"],
            color="tab:blue",
            alpha=0.6,
            label="Luong",
        )
        axs2.tick_params(axis="y", labelcolor="tab:blue")

        lines, labels = axs.get_legend_handles_labels()
        lines2, labels2 = axs2.get_legend_handles_labels()
        axs2.legend(lines + lines2, labels + labels2, loc="lower right")

        axs.set_title(title)

        fig.set_size_inches(5.5, 3)
        fig.set_tight_layout(True)

        return fig, axs
