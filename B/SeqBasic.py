import tensorflow as tf
import tensorflow_addons as tfa
from DataProcessing import DataProcessing
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd


class Seq2SeqModel:
    """
    class: Seq2SeqModel

    This class is the main class containing a basic encoder, decoder
    model without an attention block.

    It contains all of the building, training and tuning functions.

    Inspired and written based off existing implamentations:
    https://github.com/Currie32/Spell-Checker
    https://www.tensorflow.org/text/tutorials/nmt_with_attention
    https://gist.github.com/firojalam/66bf79c3746731d85e9ea9fad7a22099


    args:
        buffer (int): buffer size for data loading/shuffling
        batch_size (int): number of setnences to run per batch
        num_examples (int): number of exampels to load in to train set
        learning_rate (int): learning rate of adam optomiser
        file_dir (str): location to save checkpoints
        train_dir (str): training set directory
        test_dir (str): test set directory
    methods:
        build_model: builds the given model
        compile_model: Compiles the given model
        train: function to train model
        test: function to test model on test set
        initialize_hidden_state: QoL function used to build and
        initalise hidden states.
        train_step: tf train step
        validation_step: tf validation step
        loss_function: calualtes masked loss
        masked_accuracy: calculates masked accuracy

    Example:
        SeqAttention()

    """

    def __init__(
        self,
        buffer: int = 100000,
        batch_size: int = 128,
        num_examples: int = 1000,
        learning_rate: float = 0.0001,
        file_dir: str = "./B/training",
        train_dir: str = "Misspelling_Corpus.csv",
        test_dir: str = "Test_Set.csv",
    ):
        self.buffer = buffer
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.learning_rate = learning_rate
        self.data_processing_inst = DataProcessing()

        script_dir = os.path.dirname(os.path.realpath(__file__))
        data_set_root = os.path.join(script_dir, "..", "Dataset", train_dir)

        test_set_root = os.path.join(script_dir, "..", "Dataset", test_dir)

        self.train_dataset, self.val_dataset, self.inp_token, self.targ_token = (
            self.data_processing_inst.call_train_val(
                num_examples, self.buffer, self.batch_size, root=data_set_root
            )
        )
        self.test_dataset = self.data_processing_inst.call_test(
            self.buffer,
            self.batch_size,
            self.inp_token,
            self.targ_token,
            root=test_set_root,
        )
        # Print vocabulary information
        for word, index in self.inp_token.word_index.items():
            print(f"Word: {word}, Index: {index}")

        example_input_batch, example_target_batch = next(iter(self.train_dataset))

        # Various model perameters
        self.vocab_inp_size = len(self.inp_token.word_index) + 1
        self.vocab_tar_size = len(self.targ_token.word_index) + 1
        self.max_length_input = example_input_batch.shape[1]
        self.max_length_output = example_target_batch.shape[1]
        self.embedding_dim = 256
        self.enc_units = 512
        self.dec_units = 512
        self.steps_per_epoch = num_examples // self.batch_size

        # Build and compile the model
        self.build_model()
        self.compile_model()
        self.optimizer = tf.keras.optimizers.Adam()

        timestamp = str(int(time.time()))
        self.checkpoint_dir = os.path.join(file_dir, timestamp)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            encoder=self.encoder_lstm,
            decoder=self.decoder_lstm,
        )

    def build_model(self):
        """
        method: build_model

        Builds the given model. Builds basic, encoder
        decoder model.

        args:
            None
        return:
            None
        Example:
            SeqBasic.build_model()
        """

        # builds the encoder
        self.encoder_embedding = tf.keras.layers.Embedding(
            self.vocab_inp_size, self.embedding_dim
        )
        self.encoder_lstm = tf.keras.layers.LSTM(
            self.enc_units, return_sequences=True, return_state=True
        )

        # builds the decoder
        self.decoder_embedding = tf.keras.layers.Embedding(
            self.vocab_tar_size, self.embedding_dim
        )
        self.decoder_lstm = tf.keras.layers.LSTM(
            self.dec_units, return_sequences=True, return_state=True
        )
        self.fc = tf.keras.layers.Dense(self.vocab_tar_size)

    def compile_model(self):
        """
        method: compile_model

        Compiles the given model. This is for the basic the encoder,
        decoder model.

        args:
            None
        return:
            None
        Example:
            SeqBasic.compile_model()
        """
        encoder_inputs = tf.keras.Input(shape=(None,))
        decoder_inputs = tf.keras.Input(shape=(None,))

        encoder_embedded = self.encoder_embedding(encoder_inputs)
        encoder_output, encoder_state_h, encoder_state_c = self.encoder_lstm(
            encoder_embedded
        )

        decoder_embedded = self.decoder_embedding(decoder_inputs)
        decoder_output, _, _ = self.decoder_lstm(
            decoder_embedded, initial_state=[encoder_state_h, encoder_state_c]
        )

        outputs = self.fc(decoder_output)

        self.model = tf.keras.Model([encoder_inputs, decoder_inputs], outputs)
        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

    def train(self, epochs: int = 5):
        """
        method: train

        Given a set number of epochs, it will train the current loaded model
        on that number of epochs on the peramter set batch size.

        args:
            epochs (int): number of epochs to train algorithm.

        return:
            None
        Example:
            SeqBasic.train()
        """

        train_loss = []
        train_acc = []
        val_losses = []
        val_accuracies = []
        epoch_time = []

        # Epoch Loop
        for epoch in range(epochs):
            start = time.time()

            enc_hidden = self.initialize_hidden_state()
            total_loss = 0
            total_accuracy = 0

            # Batch Loop
            for batch, (inp, targ) in enumerate(
                self.train_dataset.take(self.steps_per_epoch)
            ):
                batch_loss, batch_acc = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss
                total_accuracy += batch_acc

                if batch % 100 == 0:
                    print(
                        "Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                            epoch + 1, batch, batch_loss.numpy(), batch_acc.numpy()
                        )
                    )
            epoch_loss = total_loss / self.steps_per_epoch
            epoch_acc = total_accuracy / self.steps_per_epoch
            train_loss.append(epoch_loss.numpy())
            train_acc.append(epoch_acc.numpy())

            val_total_loss = 0
            val_total_accuracy = 0
            num_val_batches = 0

            for val_inp, val_targ in self.val_dataset:
                val_batch_loss, val_batch_acc, _ = self.validation_step(
                    val_inp, val_targ, enc_hidden
                )
                val_total_loss += val_batch_loss
                val_total_accuracy += val_batch_acc
                num_val_batches += 1

            val_loss = val_total_loss / num_val_batches
            val_accuracy = val_total_accuracy / num_val_batches

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)
            time_elapsed = time.time() - start
            epoch_time.append(time_elapsed)

            if (epoch + 1) % 1 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            print(
                "Epoch {} Loss {:.4f} Accuracy {:.4f}".format(
                    epoch + 1,
                    total_loss / self.steps_per_epoch,
                    total_accuracy / self.steps_per_epoch,
                )
            )
            print(
                "Validation Loss {:.4f}, Validation Accuracy {:.4f}".format(
                    val_loss, val_accuracy
                )
            )
            print("Time taken for 1 epoch {:.0f} sec\n".format(time_elapsed))

        # Test pass
        test_total_loss = 0
        test_total_accuracy = 0
        num_test_batches = 0
        test_true_labels = []
        test_pred_labels = []

        for test_inp, test_targ in self.test_dataset:
            test_batch_loss, test_batch_acc, test_batch_pred = self.validation_step(
                test_inp, test_targ, enc_hidden
            )
            test_total_loss += test_batch_loss
            test_true_labels.extend(test_targ.numpy().tolist())
            test_pred_labels.extend(test_batch_pred)
            test_total_accuracy += test_batch_acc
            num_test_batches += 1

        data_to_save = {
            "losses": train_loss,
            "accuracies": train_acc,
            "val_losses": val_losses,
            "time": epoch_time,
            "val_accuracies": val_accuracies,
            "test_total_loss": [test_total_loss.numpy()],
            "test_total_accuracy": [test_total_accuracy.numpy()],
        }
        padded_data = {key: pd.Series(value) for key, value in data_to_save.items()}

        df = pd.DataFrame(padded_data)
        self.run_timestamp = str(int(time.time()))
        df.to_csv(self.checkpoint_dir + f"/{self.run_timestamp}_metrics.csv")

        test_loss = test_total_loss / num_test_batches
        test_acc = test_total_accuracy / num_test_batches

        print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_acc))

    def initialize_hidden_state(self):
        """
        method: initialize_hidden_state

        QoL that initialises the hidden states of the encoder/decoder when called in
        the given format.

        args:
            None
        return:
            : array of hidden states of 0
        Example:
            SeqBasic.initialize_hidden_state()
        """
        return [
            tf.zeros((self.batch_size, self.enc_units)),
            tf.zeros((self.batch_size, self.enc_units)),
        ]

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        """
        method: train_step

        Tensorflow Function optomised to perform the train step pass for a given batch

        args:
            inp(): input tokens
            targ(): target toakns
            enc_hidden(): encoder hidden layer

        return:
            loss(): loss of this training step parse
            acc(): accuracy of this traing step parse
        Example:
            SeqBasic.train_step(inp, targ, enc_hidden)

        """
        loss = 0
        acc = 0

        with tf.GradientTape() as tape:
            enc_output, enc_h, enc_c = self.encoder_lstm(
                self.encoder_embedding(inp), initial_state=enc_hidden
            )

            dec_input = targ[:, :-1]
            real = targ[:, 1:]

            dec_output, _, _ = self.decoder_lstm(
                self.decoder_embedding(dec_input), initial_state=[enc_h, enc_c]
            )
            logits = self.fc(dec_output)
            loss = self.loss_function(real, logits)
            acc = self.masked_accuracy(real, logits)

        variables = (
            self.encoder_lstm.trainable_variables
            + self.decoder_lstm.trainable_variables
            + self.fc.trainable_variables
        )
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, acc

    @tf.function
    def validation_step(self, inp, targ, enc_hidden):
        """
        method: validation_step

        Tensorflow Function optomised to perform the validation/test step pass for a given batch

        args:
            inp(): input tokens
            targ(): target toakns
            enc_hidden(): encoder hidden layer

        return:
            loss(): loss of this training step parse
            acc(): accuracy of this traing step parse
            predictions(): predictions of a given input
        Example:
            SeqBasic.validation_step(inp, targ, enc_hidden)

        """
        loss = 0
        acc = 0

        enc_output, enc_h, enc_c = self.encoder_lstm(
            self.encoder_embedding(inp), initial_state=enc_hidden
        )

        dec_input = targ[:, :-1]
        real = targ[:, 1:]

        dec_output, _, _ = self.decoder_lstm(
            self.decoder_embedding(dec_input), initial_state=[enc_h, enc_c]
        )
        logits = self.fc(dec_output)
        loss = self.loss_function(real, logits)
        acc = self.masked_accuracy(real, logits)
        predictions = tf.argmax(logits, axis=-1)

        return loss, acc, predictions

    def loss_function(self, real, pred):
        """
        method: loss_function

        Utility function that calculates masked loss of a output and target


        args:
            real() : real output
            pred() : prediciton output

        return:
            loss(float): computed loss

        Example:
            SeqBasic.loss_function(real, pred)

        """
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = cross_entropy(y_true=real, y_pred=pred)

        mask = tf.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=loss.dtype)

        loss = mask * loss
        loss = tf.reduce_mean(loss)
        return loss

    def masked_accuracy(self, real, pred):
        """
        method: masked_accuracy

        Utility function that calculates masked accuracy of a given batch

        args:
            real() : real input string
            pred() : prediciton output string

        return:
            accuracy(): computed masked accuracy of given set of strings

        Example:
            SeqAttention.masked_accuracy(real, pred)

        """
        mask = tf.math.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=tf.float32)
        correct_predictions = tf.cast(
            tf.equal(real, tf.argmax(pred, axis=-1, output_type=tf.int32)),
            dtype=tf.float32,
        )
        correct_predictions *= mask
        accuracy = tf.reduce_sum(correct_predictions) / tf.maximum(
            tf.reduce_sum(mask), 1
        )
        return accuracy
