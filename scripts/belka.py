import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, LayerNormalization, Add, MultiHeadAttention, Dropout
from keras.layers import Embedding, TextVectorization, GlobalAvgPool1D
from keras import Sequential
from sklearn.model_selection import train_test_split
import os
from typing import Union
import mapply
from skfp.fingerprints import ECFPFingerprint
from rdkit import Chem
import atomInSmiles
import dask.dataframe as dd
import pyarrow as pa
import itertools
import einops


# LOSSES and METRICS >>>

@keras.saving.register_keras_serializable(package='belka', name='MultiLabelLoss')
class MultiLabelLoss(keras.losses.Loss):
    """
    * Macro- or Micro-averaged Weighted Masked Binary Focal loss.
    * Dynamic mini-batch class weights "alpha".
    * Used for binary multilabel classification.
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


@keras.saving.register_keras_serializable(package='belka', name='CategoricalLoss')
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


@keras.saving.register_keras_serializable(package='belka', name='BinaryLoss')
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


@keras.saving.register_keras_serializable(package='belka', name='MaskedAUC')
class MaskedAUC(keras.metrics.AUC):
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


# LAYERS >>>

class FPGenerator(tf.keras.layers.Layer):
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


@keras.saving.register_keras_serializable(package='belka', name='Encodings')
class Encodings(keras.layers.Layer):
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


@keras.saving.register_keras_serializable(package='belka', name='Embeddings')
class Embeddings(tf.keras.layers.Layer):
    def __init__(self, max_length: int, depth: int, input_dim: int, name: str = 'embeddings', **kwargs):
        super(Embeddings, self).__init__(name=name)
        self.depth = depth
        self.max_length = max_length
        self.input_dim = input_dim
        self.embeddings = Embedding(input_dim=input_dim, output_dim=depth, mask_zero=True)
        self.encodings = Encodings(**parameters)

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


@keras.saving.register_keras_serializable(package='belka', name='FeedForward')
class FeedForward(tf.keras.layers.Layer):
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


@keras.saving.register_keras_serializable(package='belka', name='SelfAttention')
class SelfAttention(tf.keras.layers.Layer):
    """
    * Self-Attention block with PRE-layer normalization
    * LayerNorm -> MHA -> Skip connection
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


@keras.saving.register_keras_serializable(package='belka', name='EncoderLayer')
class EncoderLayer(tf.keras.layers.Layer):
    """
    * Encoder layer with PRE-layer normalization: LayerNorm -> Self-Attention -> LayerNorm -> FeedForward.
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


# MODELS >>>

@keras.saving.register_keras_serializable(package='belka', name='SingleOutput')
class Belka(tf.keras.Model):
    def __init__(self, dropout_rate: float, mode: str, num_layers: int, vocab_size: int, **kwargs):
        super(Belka, self).__init__()

        # Arguments
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.mode = mode

        #  Layers
        self.embeddings = Embeddings(input_dim=vocab_size, name='smiles_emb', **parameters)
        self.encoder = [EncoderLayer(name='encoder_{}'.format(i), **parameters) for i in range(num_layers)]
        if mode == 'mlm':
            self.head = Dense(units=vocab_size, activation='softmax', name='smiles')
        else:
            self.head = Sequential([
                GlobalAvgPool1D(),
                Dropout(dropout_rate),
                Dense(units = 3 if mode == 'clf' else 2048, activation='sigmoid')])

    def call(self, inputs, training=False, *args, **kwargs):
        x = self.embeddings(inputs, training=training)
        for encoder in self.encoder:
            x = encoder(x, training=training)
        x = self.head(x, training=training)
        return x

    def get_config(self) -> dict:
        return {
            'mode': self.mode,
            'num_layers': self.num_layers,
            'vocab_size': self.vocab_size}

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**config)


# DATASETS >>>

def train_val_set(batch_size: int, buffer_size: int, masking_rate: float, max_length: int, mode: str, seed: int,
                  vocab_size: int,working: str, **kwargs) -> tuple:
    """
    Make train and validation datasets.
    """

    # Constants
    auto = tf.data.AUTOTUNE
    encoder = get_smiles_encoder(**parameters)

    # Helper functions >>>

    def encode_smiles(x, **kwargs) -> dict:
        """
        Encode SMILES strings.
        """
        x['smiles'] = tf.io.parse_tensor(x['smiles'], out_type=tf.string)
        x['smiles'] = tf.cast(encoder(x['smiles']), dtype=tf.int32)
        return x

    def get_model_inputs(x) -> tuple:
        """
        MLM mode: mask [mask_rate] of non-zero tokens.
            * 80% of the time: Replace with the [MSK].
            * 10% of the time: Replace with a random token.
            * 10% of the time: Keep the token unchanged.
        """

        if mode == 'mlm':

            # Get paddings mask (0 = [PAD])
            paddings_mask = tf.cast(x['smiles'] != 0, dtype=tf.float32)

            # Get random mask (1 = [MASK])
            probs = tf.stack([1.0 - masking_rate, masking_rate * 0.8, masking_rate * 0.1, masking_rate * 0.1], axis=0)
            probs = tf.expand_dims(probs, axis=0)
            probs = tf.ones_like(x['smiles'], dtype=tf.float32)[:, :4] * probs
            probs = tf.math.log(probs)
            mask = tf.multiply(
                tf.one_hot(
                    indices=tf.random.categorical(logits=probs, num_samples=tf.shape(x['smiles'])[-1]),
                    depth=4,
                    dtype=tf.float32,
                    seed=seed),
                tf.expand_dims(paddings_mask, axis=-1))
            mask = tf.cast(mask, dtype=tf.int32)

            # Compute masked inputs
            x['masked'] = tf.multiply(
                mask,
                tf.stack(
                    values=[
                        x['smiles'],
                        tf.ones_like(x['smiles']),
                        tf.random.uniform(shape=tf.shape(x['smiles']), minval=2, maxval=vocab_size + 1, dtype=tf.int32),
                        x['smiles']],
                    axis=-1))
            x['masked'] = tf.reduce_sum(x['masked'], axis=-1)
            mask = tf.reduce_sum(mask[:, :, 1:], axis=-1)

            # Set non-masked values to -1
            x['smiles'] = tf.stack(
                values=[(x['smiles'] * mask) - (1 - mask), x['smiles']],
                axis=-1)

            return x['masked'], x['smiles']

        elif mode == 'fps':
            return x['smiles'], x['ecfp']

        else:
            return x['smiles'], x['binds']

    # Read subsets
    encoder = get_smiles_encoder(**parameters)
    padded_shapes = {'smiles':  (max_length,), 'ecfp': (2048,), 'binds': (3,)}
    train = tf.data.Dataset.load(os.path.join(working, 'belka.tfr'), compression='GZIP')
    if mode == 'mlm':
        features = ['smiles']
        subsets = {
            'train': train,
            'none': None}
    elif mode == 'fps':
        features = ['smiles', 'ecfp']
        subsets = {
            'train': train.filter(lambda x: tf.not_equal(x['subset'], 1)),
            'val': tf.data.Dataset.load(os.path.join(working, 'belka_val.tfr'), compression='GZIP')}
    else:
        features = ['smiles', 'binds']
        subsets = {
            'train': train.filter(lambda x: tf.equal(x['subset'], 0)),
            'val': tf.data.Dataset.load(os.path.join(working, 'belka_val.tfr'), compression='GZIP')}

    # Preprocess subsets:  Cache -> [Repeat -> Shuffle] -> Encode SMILES -> Batch -> Get inputs
    for key in [key for key in subsets.keys() if key != 'none']:
        subset = subsets[key].map(lambda x: {key: x[key] for key in features}, num_parallel_calls=auto)
        subset = subset.cache()
        if key == 'train':
            subset = subset.repeat().shuffle(buffer_size=buffer_size)
        subset = subset.map(lambda x: encode_smiles(x), num_parallel_calls=auto)
        subset = subset.padded_batch(batch_size=batch_size, padded_shapes={
            key: padded_shapes[key] for key in features})
        subsets[key] = subset.map(lambda x: get_model_inputs(x), num_parallel_calls=auto)
    return subsets['train'], subsets['val']


# TRAIN & SUBMISSION >>>

def train_model(model: Union[str, None], epochs: int, initial_epoch: int, mode: str, model_name: str, patience: int,
                steps_per_epoch: int, validation_steps: int, working: str, **kwargs):
    """
    Train the model.
    """

    # Train/val subsets
    train, val = train_val_set(**parameters)

    # Build the model
    if model is not None:
        model = load_model(model)
    else:
        model = Belka(**parameters)
        if mode == 'mlm':
            loss = CategoricalLoss(mask=-1, **parameters)
            metrics = MaskedAUC(mask=-1, multi_label=False, num_labels=None, **parameters)
        elif mode == 'fps':
            loss = BinaryLoss(**parameters)
            metrics = MaskedAUC(mask=-1, multi_label=False, num_labels=None, **parameters)
        else:
            loss = MultiLabelLoss(macro=True, **parameters)
            metrics = MaskedAUC(mask=2, multi_label=True, num_labels=3, **parameters)
        model.compile(optimizer=tf.keras.optimizers.Adam(), loss=loss, metrics=metrics)


    # Callbacks
    suffix = {
        'mlm': '_{epoch:03d}_{loss:.4f}.model.keras',
        'fps': '_{epoch:03d}_{auc:.4f}_{val_auc:.4f}.model.keras',
        'clf': '_{epoch:03d}_{auc:.4f}_{val_auc:.4f}.model.keras'}
    model_saver = keras.callbacks.ModelCheckpoint(
        monitor='loss', mode='min',
        filepath=os.path.join(working, model_name + suffix[mode]),
        save_best_only=False,
        save_weights_only=False)
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss', mode='min', patience=patience, restore_best_weights=True)
    learning_rate = tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5, monitor='loss')
    callbacks = [model_saver, early_stopping, learning_rate]

    # Print model summary
    x, y_true = iter(train).get_next()
    y_pred = model(x)
    print(model.summary())

    # Fit the model
    validation_steps = None if mode == 'mlm' else validation_steps
    model.fit(train, epochs=epochs, initial_epoch=initial_epoch, steps_per_epoch=steps_per_epoch,
              validation_data=val, validation_steps=validation_steps, callbacks=callbacks)

    return model


def make_submission(batch_size: dict, max_length: int, model: str, working: str, **kwargs) -> None:
    """
    Make submission.
    """

    # Make train dataset >>>
    df = read_parquet(subset='test', **parameters).iloc[:1000]
    df['smiles'] = df['smiles'].mapply(lambda x: atomInSmiles.smiles_tokenizer(x))
    ds = tf.data.Dataset.from_tensor_slices(
        {'smiles': tf.ragged.constant(df['smiles'].tolist())})

    # Tokenize -> Zero-pad -> Batch -> Cast
    encoder = get_smiles_encoder(**parameters)
    ds = ds.map(lambda x: tf.cast(encoder(x['smiles']), dtype=tf.int32))
    ds = ds.padded_batch(batch_size=batch_size, padded_shapes=(max_length,))
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    # Make predictions >>>
    model = tf.saved_model.load(model)
    pred = np.zeros(shape=(0,3), dtype=np.float32)
    for batch in ds:
        pred = np.concatenate([pred, model.serve(batch)])
    print('\r')

    # Write predictions to csv
    cols = ['BRD4_pred', 'HSA_pred', 'sEH_pred']
    df[cols] = pred
    cols = [['BRD4', 'BRD4_pred'], ['HSA', 'HSA_pred'], ['sEH', 'sEH_pred']]
    df = np.concatenate([df[col].to_numpy() for col in cols], axis=0)
    df = pd.DataFrame(data=df, columns=['id', 'binds'])
    df = df.dropna().sort_values(by='id').reset_index(drop=True)
    df['id'] = df['id'].astype(int)
    df.to_csv(os.path.join(working, 'submission.csv'), index=False)

    return None


# UTILS >>>

def read_parquet(subset: str, root: str, **kwargs) -> pd.DataFrame:
    """
    Read and preprocess train/test parquet files.
    """

    # Read train set
    df = pd.read_parquet(os.path.join(root, f'{subset}.parquet'))
    df = df.rename(columns={
        'buildingblock1_smiles': 'block1',
        'buildingblock2_smiles': 'block2',
        'buildingblock3_smiles': 'block3',
        'molecule_smiles': 'smiles'})

    # Group by molecule -> get multiclass labels
    cols = ['block1', 'block2', 'block3', 'smiles']
    values = 'binds' if subset == 'train' else 'id'
    df = df.pivot(index=cols, columns='protein_name', values=values).reset_index()

    return df


def make_parquet(working: str, seed: int, **kwargs) -> None:
    """
    Make Dask DataFrame:

    * Read and shuffle dataframe.
    * Stack labels binding affinity [BRD4, HSA, sEH]. Nan mask = 2.
    * Validation split (at least one non-shared blocks).
    * Add train/validation/test indicator (0/1/2).
    * Replace [Dy] DNA-linker with [H]
    * Get ECFPs
    * Write to parquet file.

    Source of extra data: https://chemrxiv.org/engage/chemrxiv/article-details/6438943f08c86922ffeffe57
    Processed by: @chemdatafarmer @Hengck23
    """

    def validation_split(x, test: set):
        """
        Get train (0), or validation (1) indicators.
        Train subset: zero-intersection between blocks and "test" blocks.
        Validation subset: non-zero-intersection between blocks and "test" blocks.
        """

        blocks = set(x[col] for col in ['block1', 'block2', 'block3'])
        i = len(blocks.intersection(test))
        i = 0 if i == 0 else 1
        i = np.int8(i)

        return i

    def replace_linker(smiles):
        """
        Replace [Dy] linker with hydrogen.
        """
        smiles = smiles.replace('[Dy]', '[H]')
        smiles = Chem.CanonSmiles(smiles)
        return smiles

    # Iterate over subsets
    dataset = []
    for subset in ['test', 'extra', 'train']:

        # Read parquet/csv
        if subset in ['train', 'test']:
            df = read_parquet(subset=subset, **parameters)
        else:
            df = pd.read_csv(os.path.join(working, 'DNA_Labeled_Data.csv'), usecols=['new_structure', 'read_count'])
            df = df.rename(columns={'new_structure': 'smiles', 'read_count': 'binds'})

        # Stack binding affinity labels
        cols = ['BRD4', 'HSA', 'sEH']
        if subset == 'train':
            df['binds'] = np.stack([df[col].to_numpy() for col in cols], axis=-1, dtype=np.int8).tolist()
        elif subset == 'test':
            df["binds"] = np.tile(np.array([[2, 2, 2]], dtype=np.int8), reps=(df.shape[0], 1)).tolist()
        else:
            df['binds'] = df['binds'].mapply(lambda x: [2, 2, np.clip(x, a_min=0, a_max=1)])
        for col in cols:
            df = df.drop(columns=[col]) if col in df.columns else df

        # Validation split
        if subset == 'train':
            blocks = list(set(df['block1'].to_list()) | set(df['block2'].tolist()) | set(df['block3'].tolist()))
            _, val, _, _ = train_test_split(blocks, blocks, test_size=0.03, random_state=seed)
            df['subset'] = df.mapply(lambda x: validation_split(x, test=val), axis=1)
        elif subset == 'test':
            df['subset'] = 2
        else:
            df['subset'] = 0    # Use extra data only for training
        cols = ['block1', 'block2', 'block3']
        for col in cols:
            df = df.drop(columns=[col]) if col in df.columns else df

        # Replace [Dy] DNA-linker with [H]
        df['smiles_no_linker'] = df['smiles'].mapply(lambda x: replace_linker(smiles=x))

        # Append dataframe to list
        dataset.append(df)

    # Concatenate -> Shuffle -> Convert to Dask Dataframe
    df = pd.concat(dataset)
    df = df.sample(frac=1.0, ignore_index=True, random_state=seed)
    df = dd.from_pandas(data=df)
    df = df.repartition(npartitions=20)

    # Write to parquet
    df.to_parquet(os.path.join(working, 'belka.parquet'), schema={
        'smiles': pa.string(),
        'binds': pa.list_(pa.int8(), 3),
        'subset': pa.int8(),
        'smiles_no_linker': pa.string()})

    return None


def make_dataset(working: str, **kwargs) -> None:
    """
    Make TFR dataset.
    """

    def generator() -> dict:
        for row in df.itertuples(index=False, name='Row'):
            yield {
                'smiles': row.smiles,
                'smiles_no_linker': row.smiles_no_linker,
                'binds': row.binds,
                'subset': row.subset}

    def serialize_smiles(x) -> dict:
        """
        Serialize smiles to string.
        """

        x['smiles'] = tf.io.serialize_tensor(x['smiles'])
        return x

    def get_ecfp(x) -> dict:
        """
        Compute ECFP form "smiles_no_linker".
        """

        x['ecfp'] = transformer(x['smiles_no_linker'])
        x.pop('smiles_no_linker')
        return x

    # Read dataset
    df = dd.read_parquet(os.path.join(working, 'belka.parquet'))
    df = df.compute()

    # Tokenize SMILES
    df['smiles'] = df['smiles'].mapply(lambda x: atomInSmiles.smiles_tokenizer(x))

    # Write to TFRecords
    auto = tf.data.AUTOTUNE
    transformer = FPGenerator()
    ds = tf.data.Dataset.from_generator(
        generator=lambda : generator(),
        output_signature={
            'smiles': tf.TensorSpec(shape=(None,), dtype=tf.string),
            'smiles_no_linker': tf.TensorSpec(shape=(), dtype=tf.string),
            'binds': tf.TensorSpec(shape=(3,), dtype=tf.int8),
            'subset': tf.TensorSpec(shape=(), dtype=tf.int8)})
    ds = ds.map(lambda x: serialize_smiles(x))
    ds = ds.batch(batch_size=1024, num_parallel_calls=auto)
    ds = ds.map(lambda x: get_ecfp(x), num_parallel_calls=auto)
    ds = ds.unbatch()
    ds.save(os.path.join(working, 'belka.tfr'), compression='GZIP')
    return None


def get_vocab(working: str, **kwargs) -> None:
    """
    Get vocabulary for SMILES encoding.
    """

    # Read parquet
    df = dd.read_parquet(os.path.join(working, 'belka.parquet'))
    df = df.compute()

    # Tokenize SMILES -> Get list if unique tokens
    df['smiles'] = df['smiles'].mapply(lambda x: list(set(atomInSmiles.smiles_tokenizer(x))))
    vocab = np.unique(list(itertools.chain.from_iterable(df['smiles'].tolist()))).tolist()
    vocab = pd.DataFrame(data=vocab)
    vocab.to_csv(os.path.join(working, 'vocab.txt'), index=False, header=False)
    return None


def get_smiles_encoder(vocab: str, **kwargs) -> TextVectorization:
    """
    Get TextVectorization SMILES encoder.
    """

    tokenizer = TextVectorization(
        standardize=None,
        split=None,
        vocabulary=vocab)
    return tokenizer


def load_model(model: str, **kwargs) -> tf.keras.Model:
    """
    Load the model.
    """

    model = tf.keras.models.load_model(model, compile=True, custom_objects={
        'Encodings': Encodings,
        'Embeddings': Embeddings,
        'FeedForward': FeedForward,
        'SelfAttention': SelfAttention,
        'EncoderLayer': EncoderLayer,
        'MultiLabelLoss': MultiLabelLoss,
        'CategoricalLoss': CategoricalLoss,
        'BinaryLoss': BinaryLoss,
        'MaskedAUC': MaskedAUC})
    return model


def set_parameters(
        activation: str, batch_size: int, buffer_size: Union[int, float],
        depth: int,
        dropout_rate: float, epochs: int, epsilon: float, initial_epoch: int,
        masking_rate: float, max_length: int, mode: str, model: Union[str, None],
        model_name: str, num_heads: int, num_layers: int,
        patience: int, root: str, seed: int, steps_per_epoch: int, validation_steps: int, vocab: str, vocab_size: int,
        working: str) -> dict:
    """
    Set uniform parameters for the functions in the scope of the project.
    :param mode: Choose from ['clf', 'fps', 'mlm'],
    :param vocab_size: Set to N+2, where N - number of tokens ([PAD] = 0, [MASK] = 1).    :return:
    """

    inputs = {
        'activation': activation,
        'batch_size': batch_size,
        'buffer_size': int(buffer_size),
        'depth': depth,
        'dropout_rate': dropout_rate,
        'epochs': epochs,
        'epsilon': epsilon,
        'initial_epoch': initial_epoch,
        'masking_rate': masking_rate,
        'max_length': max_length,
        'mode': mode,
        'model': model,
        'model_name': model_name,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'patience': patience,
        'root': root,
        'seed': seed,
        'steps_per_epoch': steps_per_epoch,
        'validation_steps': validation_steps,
        'vocab': vocab,
        'vocab_size': vocab_size,
        'working': working}

    return inputs


# EXECUTION >>>

# Set parameters
mapply.init(n_workers=-1, progressbar=True)
parameters = set_parameters(
    root='',
    working='working',
    vocab='working/vocab.txt',
    model=None,
    mode='clf',
    model_name='belka',
    masking_rate=0.15,
    batch_size=2048, buffer_size=1e07,
    epochs=1000, initial_epoch=0, steps_per_epoch=10000, validation_steps=2000,
    max_length=128, vocab_size=43,
    depth=32, dropout_rate=0.1, num_heads=8, num_layers=4, activation='gelu',
    patience=20, epsilon=1e-07, seed=42)


df = tf.data.Dataset.load('working/belka_val.tfr', c)
print(df.cardinality())