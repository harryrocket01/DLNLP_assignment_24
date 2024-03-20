import subprocess
import sys
import os

subprocess.check_call([sys.executable, "-m", "pip", "install", "einops"])

import numpy as np

import typing
from typing import Any, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tensorflow as tf
import keras

# import tensorflow_text as tf_text
import einops


class Seq2SeqModel:
    def __init__(self):
        self.units = None
        self.lr = None
        self.model = None
        self.train, self.val = None, None
        self.target_text_processor = None
        self.context_text_processor = None
        self.checkpoint_path = "training_1/cp.ckpt"
        self.save_path = "./checkpoints/my_checkpoint"

    def set_hyper_parameters(self, units=None, lr=None):
        self.units = units if units is not None else self.units
        self.lr = lr if lr is not None else self.lr

    def set_model(self):
        self.model = SpellChecker(
            self.units, self.context_text_processor, self.target_text_processor
        )
        self.model.compile(
            optimizer="adam",
            loss=self.masked_loss,
            metrics=[self.masked_acc, self.masked_loss],
        )

    def set_dataset(
        self,
        train=None,
        val=None,
        context_text_processor=None,
        target_text_processor=None,
    ):

        self.train = train if train is not None else self.train
        self.val = val if val is not None else self.val
        self.context_text_processor = (
            context_text_processor
            if context_text_processor is not None
            else self.context_text_processor
        )
        self.target_text_processor = (
            target_text_processor
            if target_text_processor is not None
            else self.target_text_processor
        )

    def train_model(self, epochs=100):

        self.model.evaluate(self.val, steps=20, return_dict=True)

        checkpoint_dir = os.path.dirname(self.checkpoint_path)

        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.checkpoint_path, save_weights_only=True, verbose=1
        )

        history = self.model.fit(
            self.train.repeat(),
            epochs=epochs,
            steps_per_epoch=100,
            validation_data=self.val,
            validation_steps=20,
            callbacks=[cp_callback],
        )
        self.model.save_weights(self.save_path)

        return history

    def load_model(self):
        self.model.load_weights(self.save_path)

        self.model.spell_check(["hello my neme is Harry."])

    def masked_loss(self, y_true, y_pred):
        # Calculate the loss for each item in the batch.
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction="none"
        )
        loss = loss_fn(y_true, y_pred)

        # Mask off the losses on padding.
        mask = tf.cast(y_true != 0, loss.dtype)
        loss *= mask

        # Return the total.
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    def masked_acc(self, y_true, y_pred):
        # Calculate the loss for each item in the batch.
        y_pred = tf.argmax(y_pred, axis=-1)
        y_pred = tf.cast(y_pred, y_true.dtype)

        match = tf.cast(y_true == y_pred, tf.float32)
        mask = tf.cast(y_true != 0, tf.float32)

        return tf.reduce_sum(match) / tf.reduce_sum(mask)


class Encoder(tf.keras.layers.Layer):
    def __init__(self, text_processor, units):
        super(Encoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.units = units

        # The embedding layer converts tokens to vectors
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, units, mask_zero=True
        )

        # The RNN layer processes those vectors sequentially.
        self.rnn = tf.keras.layers.Bidirectional(
            merge_mode="sum",
            layer=tf.keras.layers.GRU(
                units,
                # Return the sequence and state
                return_sequences=True,
                recurrent_initializer="glorot_uniform",
            ),
        )

    def call(self, x):
        shape_checker = ShapeChecker()
        shape_checker(x, "batch s")

        # embedding layer looks up the embedding vector for each token
        x = self.embedding(x)
        shape_checker(x, "batch s units")

        # GRU processes the sequence of embeddings
        x = self.rnn(x)
        shape_checker(x, "batch s units")

        return x

    def convert_input(self, texts):
        texts = tf.convert_to_tensor(texts)
        if len(texts.shape) == 0:
            texts = tf.convert_to_tensor(texts)[tf.newaxis]
        context = self.text_processor(texts).to_tensor()
        context = self(context)
        return context


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(
            key_dim=units, num_heads=1, **kwargs
        )
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()

    def call(self, x, context):
        shape_checker = ShapeChecker()

        shape_checker(x, "batch t units")
        shape_checker(context, "batch s units")

        attn_output, attn_scores = self.mha(
            query=x, value=context, return_attention_scores=True
        )

        shape_checker(x, "batch t units")
        shape_checker(attn_scores, "batch heads t s")

        attn_scores = tf.reduce_mean(attn_scores, axis=1)
        shape_checker(attn_scores, "batch t s")
        self.last_attention_weights = attn_scores

        x = self.add([x, attn_output])
        x = self.layernorm(x)

        return x


class Decoder(tf.keras.layers.Layer):

    def __init__(self, text_processor, units):
        super(Decoder, self).__init__()
        self.text_processor = text_processor
        self.vocab_size = text_processor.vocabulary_size()
        self.word_to_id = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(), mask_token="", oov_token="[UNK]"
        )
        self.id_to_word = tf.keras.layers.StringLookup(
            vocabulary=text_processor.get_vocabulary(),
            mask_token="",
            oov_token="[UNK]",
            invert=True,
        )
        self.start_token = self.word_to_id("[START]")
        self.end_token = self.word_to_id("[END]")

        self.units = units

        # 1. embedding layer converts token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(
            self.vocab_size, units, mask_zero=True
        )

        # 2. rnn keeps track
        self.rnn = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

        # RNN output will be the query for the attention
        self.attention = CrossAttention(units)

        # fully-con layer produces the logits
        self.output_layer = tf.keras.layers.Dense(self.vocab_size)

    def call(self, context, x, state=None, return_state=False):
        shape_checker = ShapeChecker()
        shape_checker(x, "batch t")
        shape_checker(context, "batch s units")

        # set embeddings
        x = self.embedding(x)
        shape_checker(x, "batch t units")

        # Process the target sequence.
        x, state = self.rnn(x, initial_state=state)
        shape_checker(x, "batch t units")

        # the RNN output as the query for the attention
        x = self.attention(x, context)
        self.last_attention_weights = self.attention.last_attention_weights
        shape_checker(x, "batch t units")
        shape_checker(self.last_attention_weights, "batch t s")

        # Generate logit predictions
        logits = self.output_layer(x)
        shape_checker(logits, "batch t target_vocab_size")

        if return_state:
            return logits, state
        else:
            return logits

    def get_initial_state(self, context):
        batch_size = tf.shape(context)[0]
        start_tokens = tf.fill([batch_size, 1], self.start_token)
        done = tf.zeros([batch_size, 1], dtype=tf.bool)
        embedded = self.embedding(start_tokens)
        return start_tokens, done, self.rnn.get_initial_state(embedded)[0]

    def tokens_to_text(self, tokens):
        words = self.id_to_word(tokens)
        result = tf.strings.reduce_join(words, axis=-1, separator=" ")
        result = tf.strings.regex_replace(result, "^ *\[START\] *", "")
        result = tf.strings.regex_replace(result, " *\[END\] *$", "")
        return result

    def get_next_token(self, context, next_token, done, state, temperature=0.0):
        logits, state = self(context, next_token, state=state, return_state=True)

        if temperature == 0.0:
            next_token = tf.argmax(logits, axis=-1)
        else:
            logits = logits[:, -1, :] / temperature
            next_token = tf.random.categorical(logits, num_samples=1)

        done = done | (next_token == self.end_token)
        next_token = tf.where(done, tf.constant(0, dtype=tf.int64), next_token)

        return next_token, done, state


class SpellChecker(tf.keras.Model):
    @classmethod
    def add_method(cls, fun):
        setattr(cls, fun.__name__, fun)
        return fun

    def __init__(self, units, context_text_processor, target_text_processor):
        super().__init__()
        encoder = Encoder(context_text_processor, units)
        decoder = Decoder(target_text_processor, units)

        self.encoder = encoder
        self.decoder = decoder

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        logits = self.decoder(context, x)

        # TODO(b/250038731): remove this
        try:
            del logits._keras_mask
        except AttributeError:
            pass

        return logits

    def spell_check(self, texts, *, max_length=50, temperature=0.0):

        context = self.encoder.convert_input(texts)
        batch_size = tf.shape(texts)[0]

        tokens = []
        attention_weights = []
        next_token, done, state = self.decoder.get_initial_state(context)

        for _ in range(max_length):
            # get next token
            next_token, done, state = self.decoder.get_next_token(
                context, next_token, done, state, temperature
            )

            # collect all tokans
            tokens.append(next_token)
            attention_weights.append(self.decoder.last_attention_weights)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break
        # stack tokens and weight
        tokens = tf.concat(tokens, axis=-1)
        self.last_attention_weights = tf.concat(attention_weights, axis=1)

        result = self.decoder.tokens_to_text(tokens)
        print(result[0].numpy().decode())
        return result


# @title
class ShapeChecker:
    def __init__(self):
        # Keep a cache of every axis-name seen
        self.shapes = {}

    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return

        parsed = einops.parse_shape(tensor, names)

        for name, new_dim in parsed.items():
            old_dim = self.shapes.get(name, None)

            if broadcast and new_dim == 1:
                continue

            if old_dim is None:
                # If the axis name is new, add its length to the cache.
                self.shapes[name] = new_dim
                continue

            if new_dim != old_dim:
                raise ValueError(
                    f"Shape mismatch for dimension: '{name}'\n"
                    f"    found: {new_dim}\n"
                    f"    expected: {old_dim}\n"
                )
