from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding
import sys
import pandas as pd
import numpy as np
import codecs
import matplotlib.pyplot as plt
from model import seq2seq_model_builder

sys.path.append(r'C:/Users/sbhatnagar4/Desktop/encoder-decoder-keras')
from preprocessing import config as c

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

    VOCAB_SIZE = 14999
    tokenizer = Tokenizer(num_words=VOCAB_SIZE)
    tokenizer.fit_on_texts(decoder_texts + encoder_texts)
    word_index = tokenizer.word_index

    index2word = {}
    for k, v in word_index.items():
        if v < 15000:
            index2word[v] = k
        if v > 15000:
            continue

    word2index = {}
    for k, v in index2word.items():
        word2index[v] = k

    assert len(word2index) == len(index2word)

    encoder_sequences = tokenizer.texts_to_sequences(encoder_texts)
    decoder_sequences = tokenizer.texts_to_sequences(decoder_texts)
    encoder_sequences = encoder_sequences[:200]
    decoder_sequences = decoder_sequences[:200]

    VOCAB_SIZE = len(index2word) + 1
    MAX_LEN = 20

    encoder_input_data = pad_sequences(
        encoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')
    decoder_input_data = pad_sequences(
        decoder_sequences, maxlen=MAX_LEN, dtype='int32', padding='post', truncating='post')

    num_samples = len(encoder_sequences)
    decoder_output_data = np.zeros(
        (num_samples, MAX_LEN, VOCAB_SIZE), dtype="float32")
    for i, seqs in enumerate(decoder_input_data):
        for j, seq in enumerate(seqs):
            if seq > 0:
                decoder_output_data[i][j][seq] = 1.

    embeddings_index = get_glove()
    embedding_matrix = embedding_matrix_creater(
        embeddings_index, 50, word_index=word2index)

    embed_layer = Embedding(input_dim=VOCAB_SIZE,
                            output_dim=50, trainable=True,)
    embed_layer.build((None,))
    embed_layer.set_weights([embedding_matrix])

    model = seq2seq_model_builder(embed_layer, VOCAB_SIZE, MAX_LEN)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit([encoder_input_data, decoder_input_data],
                        decoder_output_data, epochs=5, validation_split=0.2)
    plot_loss(history)


if __name__ == "__main__":
    train_model()
