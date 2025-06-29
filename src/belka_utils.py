"""
Belka Transformer Utilities

This module contains custom Keras layers, losses, metrics, and model definitions
for the Belka molecular transformer pipeline.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LayerNormalization, Add, MultiHeadAttention, Dropout
from keras.layers import Embedding, TextVectorization, GlobalAvgPool1D
from keras import Sequential
from typing import Union
import einops
from skfp.fingerprints import ECFPFingerprint


# LOSSES AND METRICS

# @keras.saving.register_keras_serializable(package='belka', name='MultiLabelLoss')
class MultiLabelLoss(keras.losses.Loss):
    """
    Macro- or Micro-averaged Weighted Masked Binary Focal loss.
    Dynamic mini-batch class weights ("alpha").
    Used for binary multilabel classification.
    """

    def __init__(self, epsilon: float, macro: bool, gamma: float = 2.0, nan_mask: int = 2, name='loss', **kwargs):
        super(MultiLabelLoss, self).__init__(name=name)
        self.epsilon = epsilon
        self.gamma = gamma
        self.macro = macro
        self.nan_mask = nan_mask

    def call(self, y_true, y_pred):
        # Cast y_true to tf.int32
        y_true = tf.cast(y_true, dtype=tf.int32)

        # Compute class weights ("alpha"): Inverse of Square Root of Number of Samples
        # Compute "alpha" for each label if "macro" = True
        # Assign zero class weights to missing classes
        # Normalize: sum of sample weights = sample count per label
        freq = tf.math.bincount(
            arr=tf.transpose(y_true, perm=[1,0]) if self.macro else y_true,
            minlength=2, maxlength=2, dtype=tf.float32, axis=-1 if self.macro else 0)
        alpha = tf.where(condition=tf.equal(freq, 0.0), x=0.0, y=tf.math.rsqrt(freq))
        ax = 1 if self.macro else None
        alpha = alpha * tf.reduce_sum(freq, axis=ax, keepdims=True) / tf.reduce_sum(alpha*freq, axis=ax, keepdims=True)
        alpha = tf.reduce_sum(alpha * tf.one_hot(y_true, axis=-1, depth=2, dtype=tf.float32), axis=-1)

        # Mask and set to zero missing labels
        y_true = tf.cast(y_true, tf.float32)
        mask = tf.cast(tf.not_equal(y_true, tf.constant(self.nan_mask, tf.float32)), dtype=tf.float32)
        y_true = y_true * mask

        # Compute loss
        y_pred = tf.clip_by_value(y_pred, clip_value_min=self.epsilon, clip_value_max=1.0 - self.epsilon)
        pt = tf.add(
            tf.multiply(y_true, y_pred),
            tf.multiply(tf.subtract(1.0, y_true), tf.subtract(1.0, y_pred)))
        loss = - alpha * (1.0 - pt) ** self.gamma * tf.math.log(pt) * mask
        ax = 1 if self.macro else None
        loss = tf.divide(tf.reduce_sum(loss, axis=ax), tf.reduce_sum(alpha * mask, axis=ax))
        loss = tf.reduce_mean(loss)
        return loss

    def get_config(self):
        config = super(MultiLabelLoss, self).get_config()
        config.update({
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'macro': self.macro,
            'nan_mask': self.nan_mask})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @keras.saving.register_keras_serializable(package='belka', name='CategoricalLoss')
class CategoricalLoss(keras.losses.Loss):
    """
    Masked Categorical Focal loss.
    Dynamic mini-batch class weights ("alpha").
    Used for MLM training.
    """
    def __init__(self, epsilon: float, mask: int, vocab_size: int, gamma: float = 2.0, name='loss', **kwargs):
        super(CategoricalLoss, self).__init__(name=name)
        self.epsilon = epsilon
        self.gamma = gamma
        self.mask = mask
        self.vocab_size = vocab_size

    def call(self, y_true, y_pred):
        # Unpack y_true to masked (y_true) and unmasked (unmasked) arrays
        unmasked = y_true[:,:,1]
        y_true = y_true[:,:,0]

        # Reshape inputs
        y_true = einops.rearrange(y_true, 'b l -> (b l)')
        y_pred = einops.rearrange(y_pred, 'b l c -> (b l) c')

        # Drop non-masked from y_true
        mask = tf.not_equal(y_true, self.mask)
        y_true = tf.boolean_mask(y_true, mask)

        # Compute class weights ("alpha"): Inverse of Square Root of Number of Samples
        # Assign zero class weights to missing classes
        # Normalize: sum of sample weights = sample count
        freq = tf.math.bincount(unmasked, minlength=self.vocab_size, dtype=tf.float32)
        freq = tf.concat([tf.zeros(shape=(2,)), freq[2:]], axis=0)  # Set frequencies for [PAD], [MASK] = 0
        alpha = tf.where(condition=tf.equal(freq, 0.0), x=0.0, y=tf.math.rsqrt(freq))

        # Convert y_true to one-hot
        # Apply mask to y_pred
        y_true = tf.one_hot(y_true, depth=self.vocab_size, axis=-1, dtype=tf.float32)
        y_pred = tf.boolean_mask(y_pred, mask, axis=0)

        # Compute loss
        y_pred = tf.clip_by_value(y_pred, clip_value_min=self.epsilon, clip_value_max=1.0 - self.epsilon)
        pt = tf.add(
            tf.multiply(y_true, y_pred),
            tf.multiply(tf.subtract(1.0, y_true), tf.subtract(1.0, y_pred)))
        loss = - alpha * ((1.0 - pt) ** self.gamma) * (y_true * tf.math.log(y_pred))
        loss = tf.divide(tf.reduce_sum(loss), tf.reduce_sum(alpha * y_true))
        return loss

    def get_config(self) -> dict:
        config = super(CategoricalLoss, self).get_config()
        config.update({
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'mask': self.mask,
            'vocab_size': self.vocab_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @keras.saving.register_keras_serializable(package='belka', name='BinaryLoss')
class BinaryLoss(keras.losses.Loss):
    """
    Binary Focal loss.
    Used for FPs training.
    """
    def __init__(self, name='loss', **kwargs):
        super(BinaryLoss, self).__init__(name=name)
        self.loss = tf.keras.losses.BinaryFocalCrossentropy()

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_true = tf.reshape(y_true, shape=(-1, 1))
        y_pred = tf.reshape(y_pred, shape=(-1, 1))
        loss = self.loss(y_true, y_pred)
        return loss

    def get_config(self) -> dict:
        config = super(BinaryLoss, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @keras.saving.register_keras_serializable(package='belka', name='MaskedAUC')
class MaskedAUC(keras.metrics.AUC):
    """
    Masked AUC metric for different training modes.
    """
    def __init__(self, mode: str, mask: int, multi_label: bool, num_labels: Union[int, None], vocab_size: int,
                 name='auc', **kwargs):
        super(MaskedAUC, self).__init__(curve='PR', multi_label=multi_label, num_labels=num_labels, name=name)
        self.mode = mode
        self.multi_label = multi_label
        self.mask = mask
        self.num_labels = num_labels
        self.vocab_size = vocab_size

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.mode == 'mlm':
            # Unpack y_true to masked (y_true) and unmasked (unmasked) arrays
            unmasked = y_true[:, :, 1]
            y_true = y_true[:, :, 0]

            # Reshape inputs
            y_true = einops.rearrange(y_true, 'b l -> (b l)')
            y_pred = einops.rearrange(y_pred, 'b l c -> (b l) c')

            # Drop non-masked tokens from y_true
            mask = tf.not_equal(y_true, self.mask)
            y_true = tf.boolean_mask(y_true, mask)

            # Convert y_true to one-hot
            # Apply mask to y_pred
            y_true = tf.one_hot(y_true, depth=self.vocab_size, axis=-1, dtype=tf.float32)
            y_pred = tf.boolean_mask(y_pred, mask, axis=0)
            mask = None

        elif self.mode == 'clf':
            mask = tf.cast(tf.not_equal(y_true, self.mask), dtype=tf.float32)

        else:
            y_true = tf.reshape(y_true, shape=(-1,1))
            y_pred = tf.reshape(y_pred, shape=(-1,1))
            mask = tf.ones_like(y_pred, dtype=tf.float32)

        # Compute macro-averaged mAP
        super().update_state(y_true, y_pred, sample_weight=mask)

    def get_config(self) -> dict:
        config = super(MaskedAUC, self).get_config()
        config.update({
            'mode': self.mode,
            'multi_label': self.multi_label,
            'mask': self.mask,
            'num_labels': self.num_labels,
            'vocab_size': self.vocab_size})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# CUSTOM LAYERS

class FPGenerator(tf.keras.layers.Layer):
    """
    Layer for generating molecular fingerprints from SMILES strings.
    """
    def __init__(self, name: str = 'fingerprints', **kwargs):
        super(FPGenerator, self).__init__(name=name)
        self.transformer = ECFPFingerprint(include_chirality=True, n_jobs=-1)

    def call(self, inputs, *args, **kwargs):
        """
        Get fingerprints given SMILES string.
        """
        x = tf.py_function(
            func=self.get_fingerprints,
            inp=[inputs],
            Tout=tf.int8)
        return x

    def get_fingerprints(self, inputs):
        x = inputs.numpy().astype(str)
        x = self.transformer.transform(x)
        x = tf.constant(x, dtype=tf.int8)
        return x


# @keras.saving.register_keras_serializable(package='belka', name='Encodings')
class Encodings(keras.layers.Layer):
    """
    Positional encoding layer for transformer architecture.
    """
    def __init__(self, depth: int, max_length: int, name: str = 'encodings', **kwargs):
        super(Encodings, self).__init__(name=name)
        self.depth = depth
        self.max_length = max_length
        self.encodings = self._pos_encodings(depth=depth, max_length=max_length)

    def call(self, inputs, training=False, *args, **kwargs):
        scale = tf.ones_like(inputs) * tf.math.sqrt(tf.cast(self.depth, tf.float32))
        x = tf.multiply(inputs, scale)
        x = tf.add(x, self.encodings[tf.newaxis, :tf.shape(x)[1], :])
        return x

    @staticmethod
    def _pos_encodings(depth: int, max_length: int):
        """
        Get positional encodings of shape [max_length, depth]
        """
        positions = tf.range(max_length, dtype=tf.float32)[:, tf.newaxis]
        idx = tf.range(depth)[tf.newaxis, :]
        power = tf.cast(2 * (idx // 2), tf.float32)
        power /= tf.cast(depth, tf.float32)
        angles = 1. / tf.math.pow(10000., power)
        radians = positions * angles
        sin = tf.math.sin(radians[:, 0::2])
        cos = tf.math.cos(radians[:, 1::2])
        encodings = tf.concat([sin, cos], axis=-1)
        return encodings

    def get_config(self) -> dict:
        return {
            'depth': self.depth,
            'max_length': self.max_length,
            'name': self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @keras.saving.register_keras_serializable(package='belka', name='Embeddings')
class Embeddings(tf.keras.layers.Layer):
    """
    Embedding layer with positional encodings.
    """
    def __init__(self, max_length: int, depth: int, input_dim: int, name: str = 'embeddings', **kwargs):
        super(Embeddings, self).__init__(name=name)
        self.depth = depth
        self.max_length = max_length
        self.input_dim = input_dim
        self.embeddings = Embedding(input_dim=input_dim, output_dim=depth, mask_zero=True)
        self.encodings = Encodings(depth=depth, max_length=max_length)

    def build(self, input_shape):
        self.embeddings.build(input_shape=input_shape)
        super().build(input_shape)

    def compute_mask(self, *args, **kwargs):
        return self.embeddings.compute_mask(*args, **kwargs)

    def call(self, inputs, training=False, *args, **kwargs):
        x = self.embeddings(inputs)
        x = self.encodings(x)
        return x

    def get_config(self) -> dict:
        return {
            'depth': self.depth,
            'input_dim': self.input_dim,
            'max_length': self.max_length,
            'name': self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @keras.saving.register_keras_serializable(package='belka', name='FeedForward')
class FeedForward(tf.keras.layers.Layer):
    """
    Feed-forward network with pre-layer normalization.
    """
    def __init__(self, activation: str, depth: int, dropout_rate: float, epsilon: float, name: str = 'ffn', **kwargs):
        super(FeedForward, self).__init__(name=name)
        self.activation = activation
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.norm = LayerNormalization(epsilon=epsilon)
        self.dense1 = Dense(units=int(depth * 2), activation=activation)
        self.dense2 = Dense(units=depth)
        self.dropout = Dropout(rate=dropout_rate)
        self.add = Add()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False, *args, **kwargs):
        x = self.norm(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.add([x, inputs])
        return x

    def get_config(self) -> dict:
        return {
            'activation': self.activation,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'name': self.name}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @keras.saving.register_keras_serializable(package='belka', name='SelfAttention')
class SelfAttention(tf.keras.layers.Layer):
    """
    Self-Attention block with PRE-layer normalization
    LayerNorm -> MHA -> Skip connection
    """
    def __init__(self, causal: bool, depth: int, dropout_rate: float, epsilon: float, max_length: int, num_heads: int,
                 name: str = 'self_attention', **kwargs):
        super(SelfAttention, self).__init__(name=name)
        self.causal = causal
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.max_length = max_length
        self.num_heads = num_heads
        self.supports_masking = True
        self.norm = LayerNormalization(epsilon=epsilon)
        self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=depth, dropout=dropout_rate)
        self.add = Add()

    def build(self, input_shape):
        self.mha.build(input_shape=[input_shape, input_shape])
        super().build(input_shape)

    def call(self, inputs, training=False, *args, **kwargs):
        # Compute attention mask
        mask = tf.cast(inputs._keras_mask, dtype=tf.float32)
        m1 = tf.expand_dims(mask, axis=2)
        m2 = tf.expand_dims(mask, axis=1)
        mask = tf.cast(tf.linalg.matmul(m1, m2), dtype=tf.bool)

        # Compute outputs
        x = self.norm(inputs)
        x = self.mha(
            query=x,
            value=x,
            use_causal_mask=self.causal,
            training=training,
            attention_mask=mask)
        x = self.add([x, inputs])

        return x

    def get_config(self) -> dict:
        return {
            'causal': self.causal,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'max_length': self.max_length,
            'name': self.name,
            'num_heads': self.num_heads}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# @keras.saving.register_keras_serializable(package='belka', name='EncoderLayer')
class EncoderLayer(tf.keras.layers.Layer):
    """
    Encoder layer with PRE-layer normalization: LayerNorm -> Self-Attention -> LayerNorm -> FeedForward.
    """
    def __init__(self, activation: str, depth: int, dropout_rate: float, epsilon: float, max_length: int,
                 num_heads: int, name: str = 'encoder_layer', **kwargs):
        super(EncoderLayer, self).__init__(name=name)
        self.activation = activation
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.max_length = max_length
        self.num_heads = num_heads
        self.supports_masking = True
        self.self_attention = SelfAttention(causal=False, depth=depth, dropout_rate=dropout_rate, epsilon=epsilon,
                                            max_length=max_length, num_heads=num_heads)
        self.ffn = FeedForward(activation=activation, depth=depth, dropout_rate=dropout_rate, epsilon=epsilon)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False, *args, **kwargs):
        x = self.self_attention(inputs, training=training)
        x = self.ffn(x, training=training)
        return x

    def get_config(self) -> dict:
        return {
            'activation': self.activation,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'max_length': self.max_length,
            'name': self.name,
            'num_heads': self.num_heads}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# MODEL DEFINITION

# @keras.saving.register_keras_serializable(package='belka', name='Belka')
class Belka(tf.keras.Model):
    """
    Belka transformer model for molecular binding affinity prediction.
    Supports three modes: MLM (Masked Language Model), FPS (Fingerprint prediction), CLF (Classification).
    """
    def __init__(self, activation: str, depth: int, dropout_rate: float, epsilon: float, max_length: int,
                 mode: str, num_heads: int, num_layers: int, vocab_size: int, **kwargs):
        super(Belka, self).__init__()

        # Store parameters
        self.activation = activation
        self.depth = depth
        self.dropout_rate = dropout_rate
        self.epsilon = epsilon
        self.max_length = max_length
        self.mode = mode
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.vocab_size = vocab_size

        # Layers
        self.embeddings = Embeddings(
            input_dim=vocab_size, 
            depth=depth, 
            max_length=max_length,
            name='smiles_emb')
        
        self.encoder = [
            EncoderLayer(
                activation=activation,
                depth=depth,
                dropout_rate=dropout_rate,
                epsilon=epsilon,
                max_length=max_length,
                num_heads=num_heads,
                name=f'encoder_{i}') 
            for i in range(num_layers)]
        
        if mode == 'mlm':
            self.head = Dense(units=vocab_size, activation='softmax', name='smiles')
        else:
            self.head = Sequential([
                GlobalAvgPool1D(),
                Dropout(dropout_rate),
                Dense(units=3 if mode == 'clf' else 2048, activation='sigmoid')])

    def call(self, inputs, training=False, *args, **kwargs):
        x = self.embeddings(inputs, training=training)
        for encoder in self.encoder:
            x = encoder(x, training=training)
        x = self.head(x, training=training)
        return x

    def get_config(self) -> dict:
        return {
            'activation': self.activation,
            'depth': self.depth,
            'dropout_rate': self.dropout_rate,
            'epsilon': self.epsilon,
            'max_length': self.max_length,
            'mode': self.mode,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


def load_model(model_path: str, **kwargs) -> tf.keras.Model:
    """
    Load a trained Belka model with custom objects.
    
    Args:
        model_path: Path to the saved model
        
    Returns:
        Loaded Keras model
    """
    custom_objects = {
        'Encodings': Encodings,
        'Embeddings': Embeddings,
        'FeedForward': FeedForward,
        'SelfAttention': SelfAttention,
        'EncoderLayer': EncoderLayer,
        'Belka': Belka,
        'MultiLabelLoss': MultiLabelLoss,
        'CategoricalLoss': CategoricalLoss,
        'BinaryLoss': BinaryLoss,
        'MaskedAUC': MaskedAUC}
    
    model = tf.keras.models.load_model(model_path, compile=True, custom_objects=custom_objects)
    return model