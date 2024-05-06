"""
The following code was written as the final project for 
ELEC0141 Deep Learning for Natural Language Processing

Author: Harry R J Softley-Graham
Date: Jan-May 2024

"""

import tensorflow as tf
import tensorflow_addons as tfa
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


class Encoder(tf.keras.Model):
    """
    class: Encoder

    Encoder, based off of tensorflow basic Encoder.
    args:
        vocab_size: intiger size of vocab
        embedding_dim : intiger size of embedding dimentions
        enc_units : initiger size of number of encoding units
        batch_sz : intiger size of batch size
        decoder_cell: Type of cell used in the decoder, one of "LSTM", "GRU", or "RNN"

    methods:
        call
        initialize_hidden_state

    Example:
        Encoder(10,20,30,50,10,)
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        enc_units: int,
        batch_sz: int,
        encoder_cell: str = "LSTM",
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
        """
        method: call

        Returns the top hidden and cell states of the top layer of encoder

        args:
            x: Input tensor
            hidden: Initial hidden state

        return:
            O: Output Tensor
            h: hidden state
            c cell state

        Example:
            Encoder.call()

        """

        x = self.embedding(x)
        output, h, c = self.encoder_layer(x, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        """
        method: initialize_hidden_state

        initialises the hidden states of the encoder

        args:
            None
        return:
            arr: inital hidden state of 0
        Example:
            Encoder.call(initialize_hidden_state)

        """

        return [
            tf.zeros((self.batch_sz, self.enc_units)),
            tf.zeros((self.batch_sz, self.enc_units)),
        ]


class Decoder(tf.keras.Model):
    """
    class: Decoder

    This class represents the decoder component of a sequence-to-sequence model.

    args:
        vocab_size (int): size of the vocabulary
        embedding_dim (int): dimension of the word embeddings
        dec_units (int): number of units in the decoder RNN
        batch_sz (int): batch size
        max_length_input (int):  maximum length of input sequence
        max_length_output (int): maximum length of output sequence
        attention_type (int): type of attention mechanism, either "luong" or "bahdanau"
        decoder_cell (str): type of cell used in the decoder, one of "LSTM", "GRU", or "RNN"

    methods:
        build_rnn_cell: method to build the RNN cell with attention mechanism
        build_initial_state: method to build the initial state of the decoder
        call: method to call the decoder with input sequences and initial states

    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        dec_units: int,
        batch_sz: int,
        max_length_input: int,
        max_length_output: int,
        attention_type="luong",
        decoder_cell="LSTM",
    ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output

        # embedding Layer
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

        # define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell, sampler=self.sampler, output_layer=self.fc
        )

    def build_rnn_cell(self, batch_sz):
        """
        method: call

            builds RNN cells of the decoder

        args:
            batch_sz: batch size
            hidden: Initial hidden state

        return:
            O: Output Tensor
            h: hidden state
            c cell state

        Example:
            Encoder.build_rnn_cell()

        """
        rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnn_cell,
            self.attention_mechanism,
            attention_layer_size=self.dec_units,
        )
        return rnn_cell

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        """
        method: build_initial_state

        builds inital states of the decoder

        args:
            batch_sz: batch size
            encoder_state: encoder states
            Dtype: data type

        return:
            decoder_initial_state: inital states of the decoder

        Example:
            Encoder.build_initial_state()

        """

        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_sz, dtype=Dtype
        )
        decoder_initial_state = decoder_initial_state.clone(cell_state=encoder_state)
        return decoder_initial_state

    def call(self, inputs, initial_state):
        """
        method: call

        Returns final state of the decoder

        args:
            inputs: Input tensor
            initial_state: Initial hidden state

        return:
            outputs: Output Tensor

        Example:
            Encoder.call()
        """

        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(
            x,
            initial_state=initial_state,
            sequence_length=self.batch_sz * [self.max_length_output - 1],
        )
        return outputs
