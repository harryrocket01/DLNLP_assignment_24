from DataProcessing import DataProcessing
from EncoderDecoder import *


import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.get_logger().setLevel("ERROR")  # or tf.logging.set_verbosity(tf.logging.ERROR)


import tensorflow_addons as tfa

from sklearn.model_selection import train_test_split

import os
import time


class SeqAttention:

    def __init__(
        self,
        buffer=100000,
        batch_size=128,
        num_examples=1000,
        checkpoint_dir="./B/training_checkpoints",
    ):

        self.buffer = buffer
        self.batch_size = batch_size
        self.num_examples = num_examples

        self.data_processing_inst = DataProcessing()
        self.train_dataset, self.val_dataset, self.inp_lang, self.targ_lang = (
            self.data_processing_inst.call(num_examples, self.buffer, self.batch_size)
        )

        for word, index in self.inp_lang.word_index.items():
            print(f"Word: {word}, Index: {index}")

        # check if needed
        example_input_batch, example_target_batch = next(iter(self.train_dataset))
        example_input_texts = [
            self.inp_lang.sequences_to_texts([sequence.numpy()])[0]
            for sequence in example_input_batch
        ]

        """### Some important parameters"""
        self.vocab_inp_size = len(self.inp_lang.word_index) + 1
        self.vocab_tar_size = len(self.targ_lang.word_index) + 1
        self.max_length_input = example_input_batch.shape[1]
        self.max_length_output = example_target_batch.shape[1]

        self.embedding_dimentions = 256
        self.units = 1024
        self.steps_per_epoch = num_examples // self.batch_size

        #####

        ## Test Encoder Stack

        self.encoder = Encoder(
            self.vocab_inp_size,
            self.embedding_dimentions,
            self.units,
            self.batch_size,
            encoder_cell="LSTM",
        )

        # check if needed
        sample_hidden = self.encoder.initialize_hidden_state()
        sample_output, sample_h, sample_c = self.encoder(
            example_input_batch, sample_hidden
        )

        self.decoder = Decoder(
            self.vocab_tar_size,
            self.embedding_dimentions,
            self.units,
            self.batch_size,
            self.max_length_input,
            self.max_length_output,
        )

        # check if needed
        sample_x = tf.random.uniform((self.batch_size, self.max_length_output))
        self.decoder.attention_mechanism.setup_memory(sample_output)
        initial_state = self.decoder.build_initial_state(
            self.batch_size, [sample_h, sample_c], tf.float32
        )
        sample_decoder_outputs = self.decoder(sample_x, initial_state)

        """## Define the optimizer and the loss function"""
        # selected optimizer
        self.optimizer = tf.keras.optimizers.Adam()

        """## Checkpoints (Object-based saving)"""

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder
        )

        """## One train_step operations"""

    def train(self):
        """## Train the model"""

        EPOCHS = 5

        # Lists to store additional variables
        losses = []
        accuracies = []
        val_losses = []
        val_accuracies = []

        for epoch in range(EPOCHS):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            total_accuracy = 0

            for batch, (inp, targ) in enumerate(
                self.train_dataset.take(self.steps_per_epoch)
            ):
                batch_loss, batch_acc = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss
                total_accuracy += batch_acc

                # Store batch losses and accuracies
                losses.append(batch_loss.numpy())
                accuracies.append(batch_acc.numpy())

                if batch % 100 == 0:
                    print(
                        "Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
                            epoch + 1, batch, batch_loss.numpy(), batch_acc.numpy()
                        )
                    )

            # Validation pass
            val_total_loss = 0
            val_total_accuracy = 0
            num_val_batches = 0

            for val_inp, val_targ in self.val_dataset:
                val_batch_loss, val_batch_acc = self.validation_step(
                    val_inp, val_targ, enc_hidden
                )
                val_total_loss += val_batch_loss
                val_total_accuracy += val_batch_acc
                num_val_batches += 1

            val_loss = val_total_loss / num_val_batches
            val_accuracy = val_total_accuracy / num_val_batches

            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            # Saving (checkpoint) the model every 2 epochs
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
            print("Time taken for 1 epoch {} sec\n".format(time.time() - start))

        # After training, you can process and analyze the recorded metrics as needed
        # For example, print the losses, accuracies, validation losses, and validation accuracies
        print("Train Losses:", losses)
        print("Train Accuracies:", accuracies)
        print("Validation Losses:", val_losses)
        print("Validation Accuracies:", val_accuracies)

    def test(self):
        """## Use tf-addons BasicDecoder for decoding"""

        """## Restore the latest checkpoint and test"""

        # restoring the latest checkpoint in checkpoint_dir
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))

        self.spellcheck("aa batteries maintain the settings if the power ever gos off.")

        self.spellcheck("aardvark females appear tobe come ointo season once per yaer.")

        self.spellcheck("mosy aardvarks have lnog snouts.")

        self.spellcheck("abdominal paine is relieved by defecation.")

        """## Use tf-addons BeamSearchDecoder

        """

        self.beam_spellcheck("mosy aardvarks have lnog snouts.")

        self.beam_spellcheck("abdominal paine is relieved by defecation.")

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
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

        return loss, acc

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
        sentence = self.data_processing_inst.sentence_processing(sentence)

        # Tokenize input sentence at the character level
        inputs = [self.inp_lang.word_index[char] for char in sentence]
        inputs = tf.keras.preprocessing.sequence.pad_sequences(
            [inputs], maxlen=self.max_length_input, padding="post"
        )
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]
        result = ""

        enc_start_state = [
            tf.zeros((inference_batch_size, self.units)),
            tf.zeros((inference_batch_size, self.units)),
        ]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

        dec_h = enc_h
        dec_c = enc_c

        start_tokens = tf.fill([inference_batch_size], self.targ_lang.word_index["<"])
        end_token = self.targ_lang.word_index[">"]

        greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

        # Instantiate BasicDecoder object
        decoder_instance = tfa.seq2seq.BasicDecoder(
            cell=self.decoder.rnn_cell,
            sampler=greedy_sampler,
            output_layer=self.decoder.fc,
        )

        # Setup Memory and inital state in decoder
        self.decoder.attention_mechanism.setup_memory(enc_out)
        decoder_initial_state = self.decoder.build_initial_state(
            inference_batch_size, [enc_h, enc_c], tf.float32
        )
        decoder_embedding_matrix = self.decoder.embedding.variables[0]

        # decode
        outputs, _, _ = decoder_instance(
            decoder_embedding_matrix,
            start_tokens=start_tokens,
            end_token=end_token,
            initial_state=decoder_initial_state,
        )
        return outputs.sample_id.numpy()

    def spellcheck(self, sentence):
        result = self.evaluate_sentence(sentence)
        result_text = self.targ_lang.sequences_to_texts(result)
        print("Input: %s" % (sentence))
        print("Predicted translation: {}".format(result_text))

    def beam_evaluate_sentence(self, sentence, beam_width=3):
        sentence = self.data_processing_inst.sentence_processing(sentence)

        inputs = [self.inp_lang.word_index[char] for char in sentence]
        inputs = tf.keras.preprocessing.sequence.pad_sequences(
            [inputs], maxlen=self.max_length_input, padding="post"
        )
        inputs = tf.convert_to_tensor(inputs)
        inference_batch_size = inputs.shape[0]
        result = ""

        enc_start_state = [
            tf.zeros((inference_batch_size, self.units)),
            tf.zeros((inference_batch_size, self.units)),
        ]
        enc_out, enc_h, enc_c = self.encoder(inputs, enc_start_state)

        dec_h = enc_h
        dec_c = enc_c

        start_tokens = tf.fill([inference_batch_size], self.targ_lang.word_index["<"])
        end_token = self.targ_lang.word_index[">"]

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
        result, beam_scores = self.beam_evaluate_sentence(sentence)
        print(result.shape, beam_scores.shape)
        for beam, score in zip(result, beam_scores):
            print(beam.shape, score.shape)
            output = self.targ_lang.sequences_to_texts(beam)
            output = [a[: a.index(">")] for a in output]
            beam_score = [a.sum() for a in score]
            print("Input: %s" % (sentence))
            for i in range(len(output)):
                print(
                    "{} Predicted Sentence: {}  {}".format(
                        i + 1, output[i], beam_score[i]
                    )
                )


inst = SeqAttention()
SeqAttention().train()
SeqAttention().test()
