"""
Continuous-time event model.

This module defines the shared neural architecture used for General Cancer (GC)
self-supervised pretraining and fold-specific Aplasia (AP) and Neutropenic Fever (NF) fine-tuning.

Each clinical event is represented by:
    - event time,
    - numerical measurement value,
    - variable identity.

The architecture:
1. Encodes continuous time and values using Continuous Value Embedding (CVE).
2. Encodes variable identity using a learned embedding table.
3. Adds the three embeddings to form an event-triplet representation.
4. Applies padding-aware multi-head transformer encoding.
5. Aggregates event representations using global attention pooling.
6. Produces a dense output used by the pretraining or downstream fine-tuning
   pipelines.

Public model input order is always:
    [times, values, variables]

Variable ID 0 is reserved for padding. IDs 1--102 correspond to the shared
fixed vocabulary of 100 clinical variables plus Age and Gender.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import nn
from tensorflow.keras import Model
from tensorflow.keras.layers import Add, Dense, Embedding, Input, Lambda, Layer
from tensorflow.python.ops import array_ops


class CVE(Layer):
    """Map a continuous scalar time or value sequence to dense embeddings."""

    def __init__(self, hid_units: int, output_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hid_units = hid_units
        self.output_dim = output_dim

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create scalar-to-hidden and hidden-to-embedding weights."""
        self.W1 = self.add_weight(
            name="CVE_W1",
            shape=(1, self.hid_units),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b1 = self.add_weight(
            name="CVE_b1",
            shape=(self.hid_units,),
            initializer="zeros",
            trainable=True,
        )
        self.W2 = self.add_weight(
            name="CVE_W2",
            shape=(self.hid_units, self.output_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """Apply scalar -> hidden tanh -> output embedding transformation."""
        expanded = K.expand_dims(inputs, axis=-1)
        hidden = K.tanh(K.bias_add(K.dot(expanded, self.W1), self.b1))
        return K.dot(hidden, self.W2)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Return [batch, sequence_length, output_dim]."""
        return input_shape + (self.output_dim,)

    def get_config(self) -> dict[str, Any]:
        """Return serializable layer configuration."""
        config = super().get_config()
        config.update(
            {
                "hid_units": self.hid_units,
                "output_dim": self.output_dim,
            }
        )
        return config


class Attention(Layer):
    """Compute padding-aware global attention weights over event embeddings."""

    def __init__(self, hid_dim: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.hid_dim = hid_dim

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create attention projection and context-vector weights."""
        embedding_dim = input_shape[-1]

        self.W = self.add_weight(
            shape=(embedding_dim, self.hid_dim),
            name="Att_W",
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.hid_dim,),
            name="Att_b",
            initializer="zeros",
            trainable=True,
        )
        self.u = self.add_weight(
            shape=(self.hid_dim, 1),
            name="Att_u",
            initializer="glorot_uniform",
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        mask: tf.Tensor,
        mask_value: float = -1e30,
    ) -> tf.Tensor:
        """
        Compute attention weights, assigning negligible probability to padding.

        Parameters
        ----------
        inputs:
            Contextualized event embeddings with shape [batch, sequence, d].
        mask:
            Binary event mask with shape [batch, sequence], where 1 denotes a
            real event and 0 denotes padding.
        """
        scores = K.dot(K.tanh(K.bias_add(K.dot(inputs, self.W), self.b)), self.u)

        expanded_mask = K.cast(
            K.expand_dims(mask, axis=-1),
            K.floatx(),
        )
        scores = expanded_mask * scores + (1.0 - expanded_mask) * mask_value

        return K.softmax(scores, axis=-2)

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Return one attention weight per event position."""
        return input_shape[:-1] + (1,)

    def get_config(self) -> dict[str, Any]:
        """Return serializable layer configuration."""
        config = super().get_config()
        config.update({"hid_dim": self.hid_dim})
        return config


class Transformer(Layer):
    """
    Padding-aware multi-head transformer encoder.

    Each block contains multi-head self-attention, residual connections, layer
    normalization, a two-layer feed-forward network, and dropout.
    """

    def __init__(
        self,
        N: int = 2,
        h: int = 8,
        dk: int | None = None,
        dv: int | None = None,
        dff: int | None = None,
        dropout: float = 0.0,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.N = N
        self.h = h
        self.dk = dk
        self.dv = dv
        self.dff = dff
        self.dropout = dropout

        # Retained from the original implementation for historical behavior.
        self.epsilon = K.epsilon() * K.epsilon()

    def build(self, input_shape: tf.TensorShape) -> None:
        """Create weights for all Transformer blocks."""
        embedding_dim = input_shape[-1]

        if self.dk is None:
            self.dk = embedding_dim // self.h
        if self.dv is None:
            self.dv = embedding_dim // self.h
        if self.dff is None:
            self.dff = 2 * embedding_dim

        self.Wq = self.add_weight(
            shape=(self.N, self.h, embedding_dim, self.dk),
            name="Wq",
            initializer="glorot_uniform",
            trainable=True,
        )
        self.Wk = self.add_weight(
            shape=(self.N, self.h, embedding_dim, self.dk),
            name="Wk",
            initializer="glorot_uniform",
            trainable=True,
        )
        self.Wv = self.add_weight(
            shape=(self.N, self.h, embedding_dim, self.dv),
            name="Wv",
            initializer="glorot_uniform",
            trainable=True,
        )
        self.Wo = self.add_weight(
            shape=(self.N, self.dv * self.h, embedding_dim),
            name="Wo",
            initializer="glorot_uniform",
            trainable=True,
        )

        self.W1 = self.add_weight(
            shape=(self.N, embedding_dim, self.dff),
            name="W1",
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b1 = self.add_weight(
            shape=(self.N, self.dff),
            name="b1",
            initializer="zeros",
            trainable=True,
        )
        self.W2 = self.add_weight(
            shape=(self.N, self.dff, embedding_dim),
            name="W2",
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b2 = self.add_weight(
            shape=(self.N, embedding_dim),
            name="b2",
            initializer="zeros",
            trainable=True,
        )

        self.gamma = self.add_weight(
            shape=(2 * self.N,),
            name="gamma",
            initializer="ones",
            trainable=True,
        )
        self.beta = self.add_weight(
            shape=(2 * self.N,),
            name="beta",
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(
        self,
        inputs: tf.Tensor,
        mask: tf.Tensor,
        training: bool | tf.Tensor | None = None,
        mask_value: float = -1e-30,
    ) -> tf.Tensor:
        """
        Encode event embeddings through N transformer blocks.
        The implementation uses `-1e-30` for masked attention logits.
        """
        if training is None:
            training = True

        attention_mask = K.cast(K.expand_dims(mask, axis=-2), K.floatx())
        outputs = inputs

        for block_index in range(self.N):
            head_outputs = []

            for head_index in range(self.h):
                queries = K.dot(outputs, self.Wq[block_index, head_index, :, :])
                keys = K.permute_dimensions(
                    K.dot(outputs, self.Wk[block_index, head_index, :, :]),
                    (0, 2, 1),
                )
                values = K.dot(outputs, self.Wv[block_index, head_index, :, :])

                attention_scores = K.batch_dot(queries, keys)
                attention_scores = (
                    attention_mask * attention_scores
                    + (1.0 - attention_mask) * mask_value
                )

                def drop_attention_scores() -> tf.Tensor:
                    dropout_mask = K.cast(
                        K.random_uniform(shape=array_ops.shape(attention_scores))
                        >= self.dropout,
                        K.floatx(),
                    )
                    return (
                        attention_scores * dropout_mask
                        + (1.0 - dropout_mask) * mask_value
                    )

                attention_scores = tf.cond(
                    tf.cast(training, tf.bool),
                    drop_attention_scores,
                    lambda: tf.identity(attention_scores),
                )

                attention_weights = K.softmax(attention_scores, axis=-1)
                head_outputs.append(K.batch_dot(attention_weights, values))

            concatenated_heads = K.concatenate(head_outputs, axis=-1)
            projected_attention = K.dot(
                concatenated_heads,
                self.Wo[block_index, :, :],
            )

            projected_attention = tf.cond(
                tf.cast(training, tf.bool),
                lambda: tf.identity(nn.dropout(projected_attention, rate=self.dropout)),
                lambda: tf.identity(projected_attention),
            )

            outputs = outputs + projected_attention
            outputs = self._layer_normalize(
                outputs,
                self.gamma[2 * block_index],
                self.beta[2 * block_index],
            )

            feed_forward = K.bias_add(
                K.dot(
                    K.relu(
                        K.bias_add(
                            K.dot(outputs, self.W1[block_index, :, :]),
                            self.b1[block_index, :],
                        )
                    ),
                    self.W2[block_index, :, :],
                ),
                self.b2[block_index, :],
            )

            feed_forward = tf.cond(
                tf.cast(training, tf.bool),
                lambda: tf.identity(nn.dropout(feed_forward, rate=self.dropout)),
                lambda: tf.identity(feed_forward),
            )

            outputs = outputs + feed_forward
            outputs = self._layer_normalize(
                outputs,
                self.gamma[2 * block_index + 1],
                self.beta[2 * block_index + 1],
            )

        return outputs

    def _layer_normalize(
        self,
        inputs: tf.Tensor,
        gamma: tf.Tensor,
        beta: tf.Tensor,
    ) -> tf.Tensor:
        """Apply the original implementation's per-event layer normalization."""
        mean = K.mean(inputs, axis=-1, keepdims=True)
        variance = K.mean(K.square(inputs - mean), axis=-1, keepdims=True)
        standard_deviation = K.sqrt(variance + self.epsilon)
        normalized = (inputs - mean) / standard_deviation
        return normalized * gamma + beta

    def compute_output_shape(self, input_shape: tf.TensorShape) -> tf.TensorShape:
        """Preserve [batch, sequence, embedding] shape."""
        return input_shape

    def get_config(self) -> dict[str, Any]:
        """Return serializable layer configuration."""
        config = super().get_config()
        config.update(
            {
                "N": self.N,
                "h": self.h,
                "dk": self.dk,
                "dv": self.dv,
                "dff": self.dff,
                "dropout": self.dropout,
            }
        )
        return config


def build_strats(
    max_len: int,
    V: int,
    d: int,
    N: int,
    he: int,
    dropout: float,
) -> Model:
    """
    Build the shared base model.

    Parameters
    ----------
    max_len:
        Fixed number of event positions per admission.
    V:
        Vocabulary size excluding padding. The workflow uses V=102.
    d:
        Event-embedding dimension.
    N:
        Number of transformer blocks.
    he:
        Number of attention heads.
    dropout:
        Transformer dropout rate.

    Returns
    -------
    tensorflow.keras.Model
        Uncompiled model with public input order [times, values, variables].
        Its dense output is used by the pretraining pipeline, while fine-tuning
        attaches a binary sigmoid classification head.
    """
    # Public input order: [times, values, variables].
    times = Input(shape=(max_len,), name="times")
    values = Input(shape=(max_len,), name="values")
    variables = Input(shape=(max_len,), dtype="int32", name="variables")

    variable_embeddings = Embedding(
        V + 1,
        d,
        name="variable_embedding",
    )(variables)

    cve_units = int(np.sqrt(d))
    value_embeddings = CVE(
        cve_units,
        d,
        name="value_cve",
    )(values)
    
    time_embeddings = CVE(
        cve_units,
        d,
        name="time_cve",
    )(times)

    event_embeddings = Add(name="triplet_fusion")(
        [variable_embeddings, value_embeddings, time_embeddings]
    )

    # Variable ID 0 denotes padding.
    event_mask = Lambda(
        lambda variable_ids: K.clip(variable_ids, 0, 1),
        name="event_padding_mask",
    )(variables)

    contextual_embeddings = Transformer(
        N=N,
        h=he,
        dk=None,
        dv=None,
        dff=None,
        dropout=dropout,
        name="transformer_encoder",
    )(event_embeddings, mask=event_mask)

    attention_weights = Attention(
        2 * d,
        name="global_attention",
    )(contextual_embeddings, mask=event_mask)

    patient_embedding = Lambda(
        lambda tensors: K.sum(tensors[0] * tensors[1], axis=-2),
        name="attention_pooling",
    )([contextual_embeddings, attention_weights])

    forecasting_output = Dense(V, name="forecast_head")(patient_embedding)

    return Model(
        inputs=[times, values, variables],
        outputs=forecasting_output,
        name="emit_strats",
    )