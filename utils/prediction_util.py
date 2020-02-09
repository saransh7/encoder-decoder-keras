from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import support.config as c
import pickle
import keras


def process_text(text):
    tokens = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(tokens, maxlen=c.MAX_LEN, dtype='int32', padding='post', truncating='post')
    return sequence

    def reply_text(text):
    sequence = process_text(text)
    encoder_model = load_model(c.encoder_model_path)
    decoder_model = load_model(c.decoder_model_path)
    answer = model.predict([sequence, sequence])
    print(answer)