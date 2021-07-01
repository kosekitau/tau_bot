# -*- coding: utf-8 -*-
import numpy as np
import math
import re
import time

# Commented out IPython magic to ensure Python compatibility.
try:
#   %tensorflow_version 2.x
except:
  pass
import tensorflow as tf

from tensorflow.keras import layers
import tensorflow_datasets as tfds

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

import pandas as pd

df = pd.read_csv("/content/drive/MyDrive/dataset/TIU/twitter/sentencepiece/original1217_train.csv", header=None)
tiu_input = df[0].to_list()
tiu_output = df[1].to_list()

tiu_input[0:2]

"""
SentencepieceTokenizerを使う場合
!pip install tensorflow_text
from tensorflow_text import  SentencepieceTokenizer
mod = tf.io.gfile.GFile("/content/drive/MyDrive/dataset/TIU/twitter/sentencepiece/sp_tiu1217.model", 'rb').read()
tokenizer = SentencepieceTokenizer(
    model=mod,
    out_type=tf.string,
    add_bos=True, 
    add_eos=True
)
tokenizer.tokenize(["朝青龍になりたい"])
"""

!pip install sentencepiece > /dev/null
import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.Load("/content/drive/MyDrive/dataset/TIU/twitter/sentencepiece/sp_tiu1217.model")
sp.encode("朝青龍になりたい")

print(sp.get_piece_size())
sp.id_to_piece([0,1,2,3,4,5])

VOCAB_SIZE_INPUT=sp.get_piece_size()
VOCAB_SIZE_OUTPUT=sp.get_piece_size()

inputs = [[1] + sp.encode(sentence) + [2]
          for sentence in tiu_input]
outputs = [[1] + sp.encode(sentence) + [2]
          for sentence in tiu_output]

print(len(inputs), len(outputs))

MAX_LENGTH = 20
idx_to_remove = [count for count, sent in enumerate(inputs) if len(sent)>MAX_LENGTH]
for idx in reversed(idx_to_remove):
  del inputs[idx]
  del outputs[idx]


idx_to_remove = [count for count, sent in enumerate(outputs) if len(sent)>MAX_LENGTH]
for idx in reversed(idx_to_remove):
  del inputs[idx]
  del outputs[idx]

  
print(len(inputs), len(outputs))

inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=MAX_LENGTH)
outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs,
                                                       value=0,
                                                       padding='post',
                                                       maxlen=MAX_LENGTH)

BATCH_SIZE = 64
BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices((inputs, outputs))
dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

class PositionalEncoding(layers.Layer):
  def __init__(self):
    super(PositionalEncoding, self).__init__()

  def get_angles(self, pos, i, d_model):
    angles = 1/np.power(10000., (2*(i//2))/np.float32(d_model)  )
    return pos*angles #[seq_length, d_model]

  def call(self, inputs):
    seq_length = inputs.shape.as_list()[-2]
    d_model = inputs.shape.as_list()[-1]
    angles = self.get_angles(np.arange(seq_length)[:, np.newaxis],
                             np.arange(d_model)[np.newaxis, :],
                             d_model)
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    pos_encoding = angles[np.newaxis, ...]
    return inputs + tf.cast(pos_encoding, tf.float32)

def scaled_dot_product_attention(quaries, keys, values, mask):
  product = tf.matmul(quaries, keys, transpose_b=True)

  keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
  scaled_product = product / tf.math.sqrt(keys_dim)

  if mask is not None:
    scaled_product += (mask*-1e9)

  attention = tf.matmul(tf.nn.softmax(scaled_product, axis=-1), values  )

  return attention

class MultiHeadAttention(layers.Layer):
  def __init__(self, nb_proj):
    super(MultiHeadAttention, self).__init__()
    self.nb_proj = nb_proj

  def build(self, input_shape):
    self.d_model = input_shape[-1]
    assert self.d_model % self.nb_proj == 0

    self.d_proj = self.d_model // self.nb_proj
    self.quary_lin = layers.Dense(units=self.d_model)
    self.key_lin = layers.Dense(units=self.d_model)
    self.value_lin = layers.Dense(units=self.d_model)

    self.final_lin = layers.Dense(units=self.d_model)

  def split_proj(self, inputs, batch_size): # inputs: [batch_size, seq_length, nb_proj, d_proj]
    shape = (batch_size,
             -1,
             self.nb_proj,
             self.d_proj)
    splited_inputs = tf.reshape(inputs, shape=shape) #[batch_size, seq_length, nb_proj, d_proj]
    return tf.transpose(splited_inputs, perm=[0, 2, 1, 3]) #[batch_size, nb_proj, seq_length, d_proj]

  def call(self, quaries, keys, values, mask):
    batch_size = tf.shape(quaries)[0]

    quaries = self.quary_lin(quaries)
    keys = self.key_lin(keys)
    values = self.value_lin(values)

    quaries = self.split_proj(quaries, batch_size)
    keys = self.split_proj(keys, batch_size)
    values = self.split_proj(values, batch_size)

    attention = scaled_dot_product_attention(quaries, keys, values, mask)
    attention = tf.transpose(attention, perm=[0,2,1,3])
    concat_attention = tf.reshape(attention, shape=(batch_size, -1, self.d_model))
    outputs = self.final_lin(concat_attention)

    return outputs

class EncoderLayer(layers.Layer):
  def __init__(self, FFN_units, nb_proj, dropout):
    super(EncoderLayer, self).__init__()
    self.FFN_units = FFN_units
    self.nb_proj = nb_proj
    self.dropout = dropout

  def build(self, input_shape):
    self.d_model = input_shape[-1]
    self.multi_head_attention = MultiHeadAttention(self.nb_proj)
    self.dropout_1 = layers.Dropout(rate=self.dropout)
    self.norm_1 = layers.LayerNormalization(epsilon=1e-6)
    
    self.dense_1 = layers.Dense(units=self.FFN_units, activation="relu")
    self.dense_2 = layers.Dense(units=self.d_model)
    self.dropout_2 = layers.Dropout(rate=self.dropout)
    self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

  def call(self, inputs, mask, training):
    attention = self.multi_head_attention(inputs,
                                          inputs,
                                          inputs,
                                          mask)
    attention = self.dropout_1(attention, training=training)
    attention = self.norm_1(attention + inputs)

    outputs = self.dense_1(attention)
    outputs = self.dense_2(outputs)
    outputs = self.dropout_2(outputs)
    outputs = self.norm_2(outputs + attention)

    return outputs

class Encoder(layers.Layer):
  def __init__(self,
               nb_layers,
               FFN_units,
               nb_proj,
               dropout, 
               vocab_size,
               d_model,
               name='encoder'):
    super(Encoder, self).__init__(name=name)
    self.nb_layers = nb_layers
    self.d_model = d_model

    self.embedding = layers.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding()
    self.dropout = layers.Dropout(rate=dropout)
    self.enc_layers = [EncoderLayer(FFN_units,
                                   nb_proj,
                                   dropout)
                      for _ in range(nb_layers)]

  def call(self, inputs, mask, training):
    outputs = self.embedding(inputs)
    outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    outputs = self.pos_encoding(outputs)
    outputs = self.dropout(outputs, training)

    for i in range(self.nb_layers):
      outpus = self.enc_layers[i](outputs, mask, training)

    return outputs

class DecoderLayer(layers.Layer):
  def __init__(self, FFN_units, nb_proj, dropout):
    super(DecoderLayer, self).__init__()
    self.FFN_units = FFN_units
    self.nb_proj = nb_proj
    self.dropout = dropout

  def build(self, input_shape):
    self.d_model = input_shape[-1]
    
    self.multi_head_attention_1 = MultiHeadAttention(self.nb_proj)
    self.dropout_1 = layers.Dropout(rate=self.dropout)
    self.norm_1 = layers.LayerNormalization(epsilon=1e-6)

    self.multi_head_attention_2 = MultiHeadAttention(self.nb_proj)
    self.dropout_2 = layers.Dropout(rate=self.dropout)
    self.norm_2 = layers.LayerNormalization(epsilon=1e-6)

    self.dense_1  = layers.Dense(units=self.FFN_units, activation="relu")

    self.dense_2 = layers.Dense(units=self.d_model)
    self.dropout_3 = layers.Dropout(rate=self.dropout)
    self.norm_3 = layers.LayerNormalization(epsilon=1e-6)
  
  def call(self, inputs, enc_outputs, mask_1, mask_2, training):
    attention = self.multi_head_attention_1(inputs, 
                                           inputs,
                                           inputs,
                                           mask_1)
    attention = self.dropout_1(attention, training)
    attention = self.norm_1(attention+inputs)

    attention_2 = self.multi_head_attention_2(attention,
                                              enc_outputs,
                                              enc_outputs,
                                              mask_2)
    attention_2 = self.dropout_2(attention_2, training)
    attention_2 = self.norm_2(attention_2 + attention)

    outputs= self.dense_1(attention_2)
    outputs= self.dense_2(outputs)
    outputs = self.dropout_3(outputs, training)
    outputs = self.norm_3(outputs+attention_2)

    return outputs

class Decoder(layers.Layer):
  def __init__(self, nb_layers,
               FFN_units,
               nb_proj,
               dropout,
               vocab_size,
               d_model,
               name='decoder'):
    super(Decoder, self).__init__(name=name)
    self.d_model=d_model
    self.nb_layers = nb_layers

    self.embedding = layers.Embedding(vocab_size, d_model)
    self.pos_encoding = PositionalEncoding()
    self.dropout = layers.Dropout(rate=dropout)

    self.dec_layers = [DecoderLayer(FFN_units,
                                   nb_proj,
                                   dropout)
                        for _ in range(nb_layers)]

  def call(self, inputs, enc_outputs, mask_1, mask_2, training):
    outputs = self.embedding(inputs)
    outputs *= tf.math.sqrt(tf.cast(self.d_model, tf.float32)) 
    outputs = self.pos_encoding(outputs)
    outputs = self.dropout(outputs, training)

    for i in range(self.nb_layers):
      outputs = self.dec_layers[i](outputs,
                                  enc_outputs,
                                  mask_1,
                                  mask_2,
                                  training)
    return outputs

class Transformer(tf.keras.Model):
  def __init__(self, 
               vocab_size_enc,
               vocab_size_dec,
               d_model,
               nb_layers,
               FFN_units,
               nb_proj,
               dropout,
               name='transformer'):
    super(Transformer, self).__init__(name=name)

    self.encoder = Encoder(nb_layers,
                           FFN_units,
                           nb_proj,
                           dropout,
                           vocab_size_enc,
                           d_model)
    self.decoder = Decoder(nb_layers,
                           FFN_units,
                           nb_proj,
                           dropout,
                           vocab_size_dec,
                           d_model)
    self.last_linear = layers.Dense(units=vocab_size_dec)
  
  def create_padding_mask(self, seq): #seq:[batch_size, seq_length]
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]

  def create_look_ahead_mask(self, seq):
    seq_len = tf.shape(seq)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return look_ahead_mask

  def call(self, enc_inputs, dec_inputs, training):
    enc_mask = self.create_padding_mask(enc_inputs)
    dec_mask_1 = tf.maximum(
        self.create_padding_mask(dec_inputs),
        self.create_look_ahead_mask(dec_inputs)
    )
    dec_mask_2 = self.create_padding_mask(enc_inputs)

    enc_outputs = self.encoder(enc_inputs, enc_mask, training)
    dec_outputs = self.decoder(dec_inputs, 
                               enc_outputs,
                               dec_mask_1,
                               dec_mask_2,
                               training)
    outputs = self.last_linear(dec_outputs)

    return outputs

def create_padding_mask(seq): #seq:[batch_size, seq_length]
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]
def create_look_ahead_mask(seq):
    seq_len = tf.shape(seq)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
    return look_ahead_mask
seq = tf.cast([[837,836,0,273,0,0,0,0]], tf.int32)
tf.maximum(create_padding_mask(seq), create_look_ahead_mask(seq))

tf.keras.backend.clear_session()

D_MODEL = 128 #512
NB_LAYERS = 4 # 6
FFN_UNITS = 512 # 2048
NB_PROJ = 8 # 8
DROPOUT = 0.1 # 0.1
transformer = Transformer(vocab_size_enc=VOCAB_SIZE_INPUT,
                          vocab_size_dec = VOCAB_SIZE_OUTPUT,
                          d_model=D_MODEL,
                          nb_layers=NB_LAYERS,
                          FFN_units=FFN_UNITS,
                          nb_proj=NB_PROJ,
                          dropout=DROPOUT)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                            reduction="none")
def loss_function(target, pred):
  mask = tf.math.logical_not(tf.math.equal(target, 0))
  loss_ = loss_object(target, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *=mask

  return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name="train_loss")
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name="train_accuracy")

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    
    self.d_model=tf.cast(d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step*(self.warmup_steps**-1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate,
                                    beta_1=0.9,
                                    beta_2=0.98,
                                    epsilon=1e-9)

checkpoint_path = "/content/drive/MyDrive/model/TIU/tensor_sp_transformer/"

ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print("Latest checkpoint restored")

EPOCHS = 10
for epoch in range(EPOCHS):
  print("Start number of epoch {}".format(epoch+1))
  start = time.time()

  train_loss.reset_states()
  train_accuracy.reset_states()

  for (batch, (enc_inputs, targets)) in enumerate(dataset):
    dec_inputs = targets[:, :-1]
    dec_outputs_real = targets[:, 1:]
    with tf.GradientTape() as tape:
      predictions = transformer(enc_inputs, dec_inputs, True)
      loss = loss_function(dec_outputs_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    #optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    #WARNING回避で追加
    optimizer.apply_gradients((grad, var) for (grad, var) in zip(gradients, transformer.trainable_variables) if grad is not None)

    train_loss(loss)
    train_accuracy(dec_outputs_real, predictions)

    if batch % 100 == 0:
      print("Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}".format(
          epoch+1, batch, train_loss.result(), train_accuracy.result()
      ))

  ckpt_save_path = ckpt_manager.save()
  print("Saving checkpoint for epoch {} at {}".format(epoch+1, ckpt_save_path))

  print("Time taken for 1 epoch: {} secs\n".format(time.time() - start))

