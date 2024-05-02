import tensorflow as tf
import tensorflow_addons as tfa
from DataProcessing import DataProcessing
import time
import os
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import pandas as pd


class Seq2SeqModel:
    def __init__(
        self,
        buffer=100000,
        batch_size=128,
        num_examples=1000,
        learning_rate=0.0001,
        file_dir="./B/training",
        train_dir="Misspelling_Corpus.csv",
        test_dir="Test_Set.csv",
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

        # Example input batch to determine parameters
        example_input_batch, example_target_batch = next(iter(self.train_dataset))

        # Some important parameters
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

        # Define optimizer
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
        # Build the encoder
        self.encoder_embedding = tf.keras.layers.Embedding(
            self.vocab_inp_size, self.embedding_dim
        )
        self.encoder_lstm = tf.keras.layers.LSTM(
            self.enc_units, return_sequences=True, return_state=True
        )

        # Build the decoder
        self.decoder_embedding = tf.keras.layers.Embedding(
            self.vocab_tar_size, self.embedding_dim
        )
        self.decoder_lstm = tf.keras.layers.LSTM(
            self.dec_units, return_sequences=True, return_state=True
        )
        self.fc = tf.keras.layers.Dense(self.vocab_tar_size)

    def compile_model(self):
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

    def train(self):
        """Train the model"""
        EPOCHS = 5

        train_loss = []
        train_acc = []
        val_losses = []
        val_accuracies = []

        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self.initialize_hidden_state()
            total_loss = 0
            total_accuracy = 0

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

            if (epoch + 1) % 2 == 0:
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
            print("Time taken for 1 epoch")

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
            "val_accuracies": val_accuracies,
            "test_total_loss": [test_total_loss.numpy()],
            "test_total_accuracy": [test_total_accuracy.numpy()],
        }
        padded_data = {key: pd.Series(value) for key, value in data_to_save.items()}

        df = pd.DataFrame(padded_data)

        df.to_csv(self.checkpoint_dir + f"/{self.run_timestamp}_metrics.csv")

        test_loss = test_total_loss / num_test_batches
        test_acc = test_total_accuracy / num_test_batches

        print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_acc))

    def initialize_hidden_state(self):
        return [
            tf.zeros((self.batch_size, self.enc_units)),
            tf.zeros((self.batch_size, self.enc_units)),
        ]

    def test(self):
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))
        while True:
            to_conv = input()
            self.spellcheck(to_conv)
            self.beam_spellcheck(to_conv)

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
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


# Example usage:
seq2seq_model = Seq2SeqModel()
seq2seq_model.train()
