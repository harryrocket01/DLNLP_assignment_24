"""
The following code was written as the final project for 
ELEC0141 Deep Learning for Natural Language Processing

Author: Harry R J Softley-Graham
Date: Jan-May 2024

"""

from DataProcessing import DataProcessing
from EncoderDecoder import *


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # or tf.logging.set_verbosity(tf.logging.ERROR)


import tensorflow_addons as tfa
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import time


class SeqAttention:
    """
    class: SeqAttention

    ~~ DESC ~~
    args:

    methods:

    Attributes:

    Example:

    """

    def __init__(
        self,
        buffer=131072,
        batch_size=128,
        num_examples=1028,
        learning_rate=0.0001,
        file_dir="./B/training",
        attention_type="luong",
        encoder_cell="LSTM",
        decoder_cell="LSTM",
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

        print(self.train_dataset)
        print(self.val_dataset)

        self.test_dataset = self.data_processing_inst.call_test(
            self.buffer,
            self.batch_size,
            self.inp_token,
            self.targ_token,
            root=test_set_root,
        )
        print(self.test_dataset)

        for word, index in self.inp_token.word_index.items():
            print(f"Word: {word}, Index: {index}")

        # check if needed
        example_input_batch, example_target_batch = next(iter(self.train_dataset))
        example_input_texts = [
            self.inp_token.sequences_to_texts([sequence.numpy()])[0]
            for sequence in example_input_batch
        ]

        # constants and variables used within making the model dynamically
        self.vocab_inp_size = len(self.inp_token.word_index) + 1
        self.vocab_tar_size = len(self.targ_token.word_index) + 1
        self.max_length_input = example_input_batch.shape[1]
        self.max_length_output = example_target_batch.shape[1]

        self.embedding_dimentions = 128
        self.units = 1024
        self.steps_per_epoch = num_examples // self.batch_size

        # Encoder & Decoder Stack
        self.encoder = Encoder(
            self.vocab_inp_size,
            self.embedding_dimentions,
            self.units,
            self.batch_size,
            encoder_cell=encoder_cell,
        )

        self.decoder = Decoder(
            self.vocab_tar_size,
            self.embedding_dimentions,
            self.units,
            self.batch_size,
            self.max_length_input,
            self.max_length_output,
            attention_type=attention_type,
            decoder_cell=decoder_cell,
        )

        # selected
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        # checkpoint directory

        self.run_timestamp = str(int(time.time()))
        self.checkpoint_dir = os.path.join(file_dir, self.run_timestamp)
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")

        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder
        )

    def train(self, epochs):
        """
        method: train

        ~~ DESC ~~

        args:

        return:

        Example:

        """

        # Lists to store additional variables
        train_loss = []
        train_acc = []
        val_losses = []
        val_accuracies = []
        epoch_time = []
        for epoch in range(epochs):
            start = time.time()
            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            total_accuracy = 0
            esimtate_flag = 0

            for batch, (inp, targ) in enumerate(
                self.train_dataset.take(self.steps_per_epoch)
            ):
                batch_loss, batch_acc = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss
                total_accuracy += batch_acc

                # Store batch losses and accuracies

                if batch % (self.num_examples // (9 * self.batch_size)) == 0:
                    esimtate_flag += 1
                    current_time = time.time() - start
                    if esimtate_flag == 1:
                        time_estimate = 0
                    else:
                        time_estimate = current_time * (12 / (esimtate_flag - 1))
                    print(
                        "Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f} - {:.0f}/{:.0f}".format(
                            epoch + 1,
                            batch,
                            batch_loss.numpy(),
                            batch_acc.numpy(),
                            current_time,
                            time_estimate,
                        )
                    )
            # Saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_prefix)

            epoch_loss = total_loss / self.steps_per_epoch
            epoch_acc = total_accuracy / self.steps_per_epoch
            train_loss.append(epoch_loss.numpy())
            train_acc.append(epoch_acc.numpy())

            # Validation pass
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

            val_losses.append(val_loss.numpy())
            val_accuracies.append(val_accuracy.numpy())

            time_elapsed = time.time() - start
            epoch_time.append(time_elapsed)

            print(
                "Epoch {} Loss {:.8f} Accuracy {:.8f}".format(
                    epoch + 1,
                    epoch_loss,
                    epoch_acc,
                )
            )
            print(
                "Validation Loss {:.8f}, Validation Accuracy {:.8f}".format(
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
            "val_accuracies": val_accuracies,
            "time": epoch_time,
            "test_total_loss": [test_total_loss.numpy()],
            "test_total_accuracy": [test_total_accuracy.numpy()],
        }
        padded_data = {key: pd.Series(value) for key, value in data_to_save.items()}

        df = pd.DataFrame(padded_data)

        df.to_csv(self.checkpoint_dir + f"/{self.run_timestamp}_metrics.csv")

        test_loss = test_total_loss / num_test_batches
        test_acc = test_total_accuracy / num_test_batches

        print("Test Loss {:.4f}, Test Accuracy {:.4f}".format(test_loss, test_acc))

    def test(self, new_dir=None):
        """
        method:

        ~~ DESC ~~

        args:

        return:

        Example:

        """
        if new_dir == None:
            current_dir = self.checkpoint_dir
        else:
            current_dir = new_dir
        try:
            self.checkpoint.restore(tf.train.latest_checkpoint(current_dir))
        except:
            print("No model to load")
            return
        # runs 20 loops of 20 inputs
        for x in range(0, 20):
            to_conv = input()
            self.spellcheck(to_conv)
            self.beam_spellcheck(to_conv)

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        """
        method: train_step

        ~~ DESC ~~

        args:

        return:

        Example:

        """

        loss = 0
        acc = 0

        with tf.GradientTape() as tape:
            enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)

            # Ignore SOS and EOS tokens (<,>)
            dec_input = targ[:, :-1]
            real = targ[:, 1:]

            self.decoder.attention_mechanism.setup_memory(enc_output)
            decoder_initial_state = self.decoder.build_initial_state(
                self.batch_size, [enc_h, enc_c], tf.float32
            )
            pred = self.decoder(dec_input, decoder_initial_state)
            logits = pred.rnn_output
            loss = self.loss_function(real, logits)
            acc = self.masked_accuracy(real, logits)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, acc

    @tf.function
    def validation_step(self, inp, targ, enc_hidden):
        """
        method: validation_step

        ~~ DESC ~~

        args:

        return:

        Example:

        """

        loss = 0
        acc = 0

        enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)

        dec_input = targ[:, :-1]
        real = targ[:, 1:]

        self.decoder.attention_mechanism.setup_memory(enc_output)
        decoder_initial_state = self.decoder.build_initial_state(
            self.batch_size, [enc_h, enc_c], tf.float32
        )
        pred = self.decoder(dec_input, decoder_initial_state)
        logits = pred.rnn_output
        loss = self.loss_function(real, logits)
        acc = self.masked_accuracy(real, logits)

        # Compute predictions
        predictions = tf.argmax(logits, axis=-1)

        return loss, acc, predictions

    def loss_function(self, real, pred):
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = cross_entropy(y_true=real, y_pred=pred)

        # calculate mask loss accuracy
        mask = tf.logical_not(tf.math.equal(real, 0))
        mask = tf.cast(mask, dtype=loss.dtype)

        loss = mask * loss
        loss = tf.reduce_mean(loss)
        return loss

    def masked_accuracy(self, real, pred):
        """
        method: masked_accuracy

        ~~ DESC ~~

        args:

        return:

        Example:

        """
        # Ignore padding tokens
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

    def evaluate_sentence(self, sentence):
        """
        method: evaluate_sentence

        ~~ DESC ~~

        args:

        return:

        Example:
        """
        sentence = self.data_processing_inst.sentence_processing(sentence)

        # Tokenize input sentence at the character level
        inputs = [self.inp_token.word_index[char] for char in sentence]
        inputs = tf.keras.preprocessing.sequence.pad_sequences(
            [inputs], maxlen=self.max_length_input, padding="post"
        )
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]

        enc_start_state = [
            tf.zeros((inference_batch_size, self.units)),
            tf.zeros((inference_batch_size, self.units)),
        ]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

        start_tokens = tf.fill([inference_batch_size], self.targ_token.word_index["<"])
        end_token = self.targ_token.word_index[">"]

        # Enables greedy sampling
        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        # use basic decoder
        decoder_instance = tfa.seq2seq.BasicDecoder(
            cell=self.decoder.rnn_cell,
            sampler=greedy_sampler,
            output_layer=self.decoder.fc,
        )
        self.decoder.attention_mechanism.setup_memory(enc_out)
        decoder_initial_state = self.decoder.build_initial_state(
            inference_batch_size, [enc_h, enc_c], tf.float32
        )
        decoder_embedding_matrix = self.decoder.embedding.variables[0]

        # decode output and return array
        outputs, _, _ = decoder_instance(
            decoder_embedding_matrix,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
        )
        return outputs.sample_id.numpy()

    def spellcheck(self, sentence):
        """
        method: spellcheck

        ~~ DESC ~~

        args:

        return:

        Example:

        """
        result = self.evaluate_sentence(sentence)
        result_text = self.targ_token.sequences_to_texts(result)
        print("Input: %s" % (sentence))
        print("Greedy: {}".format(result_text))

    def beam_evaluate_sentence(self, sentence, beam_width=3):
        """
        method: beam_evaluate_sentence

        ~~ DESC ~~

        args:

        return:

        Example:

        """
        sentence = self.data_processing_inst.sentence_processing(sentence)

        inputs = [self.inp_token.word_index[char] for char in sentence]
        inputs = tf.keras.preprocessing.sequence.pad_sequences(
            [inputs], maxlen=self.max_length_input, padding="post"
        )
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]

        enc_start_state = [
            tf.zeros((inference_batch_size, self.units)),
            tf.zeros((inference_batch_size, self.units)),
        ]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

        start_tokens = tf.fill([inference_batch_size], self.targ_token.word_index["<"])
        end_token = self.targ_token.word_index[">"]

        enc_out = tfa.seq2seq.tile_batch(enc_out, multiplier=beam_width)
        self.decoder.attention_mechanism.setup_memory(enc_out)
        print(
            "beam_with * [batch_size, max_length_input, rnn_units] :  3 * [1, 16, 1024]] :",
            enc_out.shape,
        )

        # set decoder_inital_state which is an AttentionWrapperState considering beam_width
        hidden_state = tfa.seq2seq.tile_batch([enc_h, enc_c], multiplier=beam_width)
        decoder_initial_state = self.decoder.rnn_cell.get_initial_state(
            batch_size=beam_width * inference_batch_size, dtype=tf.float32
        )
        decoder_initial_state = decoder_initial_state.clone(cell_state=hidden_state)

        # Instantiate BeamSearchDecoder
        decoder_instance = tfa.seq2seq.BeamSearchDecoder(
            self.decoder.rnn_cell, beam_width=beam_width, output_layer=self.decoder.fc
        )
        decoder_embedding_matrix = self.decoder.embedding.variables[0]

        # The BeamSearchDecoder object's call() function takes care of everything.
        outputs, final_state, sequence_lengths = decoder_instance(
            decoder_embedding_matrix,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
        )

        # Convert the shape of outputs and beam_scores to (inference_batch_size, beam_width, time_step_outputs)
        final_outputs = tf.transpose(outputs.predicted_ids, perm=(0, 2, 1))
        beam_scores = tf.transpose(
            outputs.beam_search_decoder_output.scores, perm=(0, 2, 1)
        )

        return final_outputs.numpy(), beam_scores.numpy()

    def beam_spellcheck(self, sentence):
        """
        method: beam_spellcheck

        ~~ DESC ~~

        args:

        return:

        Example:

        """
        result, score_arr = self.beam_evaluate_sentence(sentence)

        for beam, score in zip(result, score_arr):
            print(beam.shape, score.shape)
            output = self.targ_token.sequences_to_texts(beam)
            output = [i[: i.index(">")] for i in output]
            beam_score = [j.sum() for j in score]
            print("Input: %s" % (sentence))
            for i in range(len(output)):
                print(f"Beam {i + 1}: {output[i]}  {beam_score[i]}")


inst = SeqAttention(
    buffer=131072,
    batch_size=64,
    num_examples=1000,
    learning_rate=0.0001,
    file_dir="./B/training",
    attention_type="luong",
    encoder_cell="LSTM",
    decoder_cell="GRU",
    train_dir="Misspelling_Corpus.csv",
    test_dir="Test_Set.csv",
)
inst.train(epochs=2)
# inst.test()
