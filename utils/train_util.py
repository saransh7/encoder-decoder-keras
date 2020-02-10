from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sys
import pickle
import codecs
import numpy as np
import pandas as pd
import support.config as c
import matplotlib.pyplot as plt
from utils.model import seq2seq_model_builder


def get_glove():
    embeddings_index = {}
    with open(c.glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        f.close()
    return embeddings_index


def get_inputs(file_path):
    with codecs.open(file_path, "rb", encoding="utf-8", errors="ignore") as f:
        lines = f.read().split("\r\n")
        inputs = []
        for line in lines:
            data = line.split("\n")[0]
            inputs.append(data)
    return inputs


def embedding_matrix_creater(embeddings_index, embedding_dimension, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dimension))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def plot_loss(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.figure()
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


def train_model():
    decoder_texts = get_inputs(c.decoder_input)
    encoder_texts = get_inputs(c.encoder_input)
    tokenizer = Tokenizer(num_words=c.VOCAB_SIZE)
    tokenizer.fit_on_texts(decoder_texts + encoder_texts)
    word_index = tokenizer.word_index

    with open(c.tokenizer_path, 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    index2word = {}
    for k, v in word_index.items():
        if v < c.VOCAB_SIZE:
            index2word[v] = k
        if v > c.VOCAB_SIZE:
            continue

    word2index = {}
    for k, v in index2word.items():
        word2index[v] = k

    assert len(word2index) == len(index2word)

    encoder_sequences = tokenizer.texts_to_sequences(encoder_texts)
    decoder_sequences = tokenizer.texts_to_sequences(decoder_texts)
    encoder_sequences = encoder_sequences[:c.SAMPLE_LEN]
    decoder_sequences = decoder_sequences[:c.SAMPLE_LEN]

    encoder_input_data = pad_sequences(
        encoder_sequences, maxlen=c.MAX_LEN, dtype='int32', padding='post', truncating='post')
    decoder_input_data = pad_sequences(
        decoder_sequences, maxlen=c.MAX_LEN, dtype='int32', padding='post', truncating='post')
    decoder_output_data = np.zeros(
        (c.SAMPLE_LEN, c.MAX_LEN, c.VOCAB_SIZE), dtype="float32")

    for i, seqs in enumerate(decoder_input_data):
        for j, seq in enumerate(seqs):
            if seq > 0:
                decoder_output_data[i][j][seq] = 1.

    embeddings_index = get_glove()
    embedding_matrix = embedding_matrix_creater(
        embeddings_index, c.embedding_dim, word_index=word2index)
    embed_layer = Embedding(input_dim=c.VOCAB_SIZE,
                            output_dim=c.embedding_dim, trainable=True,)
    embed_layer.build((None,))
    embed_layer.set_weights([embedding_matrix])
    model, encoder_model, decoder_model = seq2seq_model_builder(
        embed_layer, c.VOCAB_SIZE, c.MAX_LEN)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit([encoder_input_data, decoder_input_data],
                        decoder_output_data, batch_size=40, epochs=1, validation_split=c.validation_split)

    model.save(c.model_path)
    encoder_model.save(c.encoder_model_path)
    decoder_model.save(c.decoder_model_path)

    if c.plot_loss:
        plot_loss(history)


if __name__ == "__main__":
    train_model()
