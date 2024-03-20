import tensorflow as tf
import tensorflow as tf
import tensorflow_addons as tfa

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import os
import time


class Seq2SeqModel:

    def __init__(self):

        self.buffer_size = 32000
        self.batch_size = 64
        self.epochs = 10

        # Let's limit the #training examples for faster training
        num_examples = 30000

        # Figure out
        self.test_data, self.val_data, self.vocab = None, None, None

        self.example_input_batch, self.example_target_batch = next(
            iter(self.input_data)
        )
        self.vocab_inp_size = len(self.vocab) + 1
        self.vocab_tar_size = len(self.vocab) + 1
        self.max_length_input = self.example_input_batch.shape[1]
        self.max_length_output = self.example_target_batch.shape[1]

        self.embedding_dim = 256
        self.units = 1024
        self.steps_per_epoch = num_examples // self.batch_size

        self.decoder = Decoder(
            self.vocab_tar_size,
            self.embedding_dim,
            self.units,
            self.batch_size,
            self.max_length_input,
            "luong",
        )
        self.encoder = Encoder(
            self.vocab_inp_size, self.embedding_dim, self.units, self.batch_size
        )
        self.optimizer = tf.keras.optimizers.Adam()

    def set_dataset(self, input_data=None, output_data=None, vocab=None):

        self.input_data = input_data if input_data is not None else self.input_data
        self.output_data = output_data if output_data is not None else self.output_data
        self.vocab = vocab if vocab is not None else self.vocab

        self.vocab_inp_size = len(self.vocab) + 1
        self.vocab_tar_size = len(self.vocab) + 1
        self.max_length_input = self.example_input_batch.shape[1]
        self.max_length_output = self.example_target_batch.shape[1]

    def test_encoder(self):
        sample_hidden = self.encoder.initialize_hidden_state()
        sample_output, sample_h, sample_c = self.encoder(
            self.example_input_batch, sample_hidden
        )
        print(
            "Encoder output shape: (batch size, sequence length, units) {}".format(
                sample_output.shape
            )
        )
        print("Encoder h vecotr shape: (batch size, units) {}".format(sample_h.shape))
        print("Encoder c vector shape: (batch size, units) {}".format(sample_c.shape))

    def test_decoder(self):
        sample_hidden = self.encoder.initialize_hidden_state()
        sample_output, sample_h, sample_c = self.encoder(
            self.example_input_batch, sample_hidden
        )

        sample_x = tf.random.uniform((self.batch_size, self.max_length_output))
        self.decoder.attention_mechanism.setup_memory(sample_output)
        initial_state = self.decoder.build_initial_state(
            self.batch_size, [sample_h, sample_c], tf.float32
        )

        sample_decoder_outputs = self.decoder(sample_x, initial_state)

        print("Decoder Outputs Shape: ", sample_decoder_outputs.rnn_output.shape)

    def model(self):

        checkpoint_dir = "./training_checkpoints"
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, encoder=self.encoder, decoder=self.decoder
        )

        for epoch in range(self.epochs):
            start = time.time()

            enc_hidden = self.encoder.initialize_hidden_state()
            total_loss = 0
            # print(enc_hidden[0].shape, enc_hidden[1].shape)

            for batch, (inp, targ) in enumerate(
                self.train_dataset.take(self.steps_per_epoch)
            ):
                batch_loss = self.train_step(inp, targ, enc_hidden)
                total_loss += batch_loss

                if batch % 100 == 0:
                    print(
                        "Epoch {} Batch {} Loss {:.4f}".format(
                            epoch + 1, batch, batch_loss.numpy()
                        )
                    )
            # saving (checkpoint) the model every 2 epochs
            if (epoch + 1) % 2 == 0:
                checkpoint.save(file_prefix=checkpoint_prefix)

            print(
                "Epoch {} Loss {:.4f}".format(
                    epoch + 1, total_loss / self.steps_per_epoch
                )
            )
            print("Time taken for 1 epoch {} sec\n".format(time.time() - start))

    @tf.function
    def train_step(self, inp, targ, enc_hidden):
        loss = 0

        with tf.GradientTape() as tape:
            enc_output, enc_h, enc_c = self.encoder(inp, enc_hidden)

            dec_input = targ[:, :-1]  # Ignore <end> token
            real = targ[:, 1:]  # ignore <start> token

            # Set the AttentionMechanism object with encoder_outputs
            self.decoder.attention_mechanism.setup_memory(enc_output)

            # Create AttentionWrapperState as initial_state for decoder
            decoder_initial_state = self.decoder.build_initial_state(
                self.batch_size, [enc_h, enc_c], tf.float32
            )
            pred = self.decoder(dec_input, decoder_initial_state)
            logits = pred.rnn_output
            loss = self.loss_function(real, logits)

        variables = self.encoder.trainable_variables + self.decoder.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def loss_function(self, real, pred):
        # real shape = (BATCH_SIZE, max_length_output)
        # pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )
        cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = cross_entropy(y_true=real, y_pred=pred)
        mask = tf.logical_not(tf.math.equal(real, 0))  # output 0 for y=0 else output 1
        mask = tf.cast(mask, dtype=loss.dtype)
        loss = mask * loss
        loss = tf.reduce_mean(loss)
        return loss


class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        ##-------- LSTM layer in Encoder ------- ##
        self.lstm_layer = tf.keras.layers.LSTM(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [
            tf.zeros((self.batch_sz, self.enc_units)),
            tf.zeros((self.batch_sz, self.enc_units)),
        ]


class Decoder(tf.keras.Model):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        dec_units,
        batch_sz,
        max_length_input,
        attention_type="luong",
    ):
        super(Decoder, self).__init__()
        self.max_length_input = max_length_input
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(
            self.dec_units,
            None,
            self.batch_sz * [max_length_input],
            self.attention_type,
        )

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell, sampler=self.sampler, output_layer=self.fc
        )

    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnn_cell,
            self.attention_mechanism,
            attention_layer_size=self.dec_units,
        )
        return rnn_cell

    def build_attention_mechanism(
        self, dec_units, memory, memory_sequence_length, attention_type="luong"
    ):
        # ------------- #
        # typ: Which sort of attention (Bahdanau, Luong)
        # dec_units: final dimension of attention outputs
        # memory: encoder hidden states of shape (batch_size, max_length_input, enc_units)
        # memory_sequence_length: 1d array of shape (batch_size) with every element set to max_length_input (for masking purpose)

        if attention_type == "bahdanau":
            return tfa.seq2seq.BahdanauAttention(
                units=dec_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
            )
        else:
            return tfa.seq2seq.LuongAttention(
                units=dec_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
            )

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_sz, dtype=Dtype
        )
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(
            x,
            initial_state=initial_state,
            sequence_length=self.batch_sz * [self.max_length_output - 1],
        )
        return outputs
