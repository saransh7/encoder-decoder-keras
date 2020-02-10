from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from utils import model
import support.config as c
import pickle
import keras


def process_text(text, tokenizer):
    tokens = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(tokens, maxlen=c.MAX_LEN, dtype='int32', padding='post', truncating='post')
    return sequence

def reply_text(text):
    # loading saved models
    encoder_model = load_model(c.encoder_model_path)
    decoder_model = load_model(c.decoder_model_path)
    with open(c.tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)

# word index
    word_index = tokenizer.word_index
    index2word = {}
    for k, v in word_index.items():
        if v < c.VOCAB_SIZE:
            index2word[v] = k
        if v > c.VOCAB_SIZE:
            continue

    sequence = process_text(text, tokenizer)
    assert sequence.shape == (1,20)
    reply = model.decode_sequence(sequence, encoder_model, decoder_model, index2word)
    print(reply)