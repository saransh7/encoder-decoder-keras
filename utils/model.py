from keras.models import Model
from keras.layers import Embedding
from keras.layers import Input, Dense, LSTM, TimeDistributed


def seq2seq_model_builder(embed_layer, VOCAB_SIZE, MAX_LEN, HIDDEN_DIM=300):
    encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    encoder_embedding = embed_layer(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    encoder_states = [ state_h , state_c ]
    decoder_inputs = Input(shape=(MAX_LEN, ))
    decoder_embedding = embed_layer(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(
        decoder_embedding, initial_state=encoder_states)
    outputs = TimeDistributed(
        Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)

    # defining inference model
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(None,))
    decoder_state_input_c = Input(shape=(None,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_LSTM(
        decoder_embedding, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    outputs = TimeDistributed(
        Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [outputs] + decoder_states)
    return model, encoder_model, decoder_model

def decode_sequence(input_seq, encoder_model, decoder_model, index2word):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, c.MAX_LEN))
    target_seq[0, 0] = 1
    stop_condition = False
    decoded_sentence = []
    counter = 0
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_word = index2word[sampled_token_index]
        print(sampled_word)
        decoded_sentence += [sampled_word]
        if (sampled_token_index == 2 or
            len(decoded_sentence) > 20):
            stop_condition = True
        target_seq[0, counter] = sampled_token_index
        states_value = [h, c]
        counter += 1

    return decoded_sentence