# This file contains all of the suplimetnry function for the creation and

import tensorflow as tf
import tensorflow_addons as tfa
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from sklearn.model_selection import train_test_split


class Encoder(tf.keras.Model):
    """
    class: Encoder

    ~~ DESC ~~
    args:

    methods:

    Attributes:

    Example:

    """

    def __init__(
        self, vocab_size, embedding_dim, enc_units, batch_sz, encoder_cell="LSTM"
    ):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.encoder_cell = encoder_cell
        if self.encoder_cell == "RNN":
            self.encoder_layer = tf.keras.layers.SimpleRNN(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform",
            )
        elif self.encoder_cell == "GRU":
            self.encoder_layer = tf.keras.layers.GRU(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform",
            )
        else:
            self.encoder_layer = tf.keras.layers.LSTM(
                self.enc_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer="glorot_uniform",
            )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, h, c = self.encoder_layer(x, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [
            tf.zeros((self.batch_sz, self.enc_units)),
            tf.zeros((self.batch_sz, self.enc_units)),
        ]


class Decoder(tf.keras.Model):
    """
    class: Encoder

    ~~ DESC ~~
    args:

    methods:

    Attributes:

    Example:

    """

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        dec_units,
        batch_sz,
        max_length_input,
        max_length_output,
        attention_type="luong",
        decoder_cell="LSTM",
    ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.fc = tf.keras.layers.Dense(vocab_size)

        if decoder_cell == "RNN":
            self.decoder_rnn_cell = tf.keras.layers.SimpleRNNCell(self.dec_units)

        elif decoder_cell == "GRU":
            self.decoder_rnn_cell = tf.keras.layers.GRUCell(self.dec_units)

        else:
            self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        mem_seq_len = self.batch_sz * [self.max_length_input]

        if self.attention_type == "bahdanau":
            # original paper
            self.attention_mechanism = tfa.seq2seq.BahdanauAttention(
                units=self.dec_units, memory=None, memory_sequence_length=mem_seq_len
            )
        else:
            # modifed paper
            self.attention_mechanism = tfa.seq2seq.LuongAttention(
                units=self.dec_units, memory=None, memory_sequence_length=mem_seq_len
            )

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
