from keras.models import Model
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed


def seq2seq_model_builder(embed_layer, VOCAB_SIZE, MAX_LEN, HIDDEN_DIM=300):

    encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    encoder_embedding = embed_layer(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)

    decoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    decoder_embedding = embed_layer(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(
        decoder_embedding, initial_state=[state_h, state_c])

    # dense_layer = Dense(VOCAB_SIZE, activation='softmax')
    outputs = TimeDistributed(
        Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)

    return model
