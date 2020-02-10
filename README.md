# encoder-decoder-keras
A keras implementaion of encoder decoder architecture for ChatBot

Demonstrates how to implement a basic word-level *sequence-to-sequence* model.

In the general case, input sequences and output sequences have different lengths (e.g. machine translation) and the entire input sequence is required in order to start predicting the target. This requires a more advanced setup, which is what people commonly refer to when mentioning "sequence to sequence models" with no further context.


## Training Mode
- A RNN acts as "encoder", processes input sequence and return its own internal state
```
    encoder_inputs = Input(shape=(MAX_LEN, ), dtype='int32',)
    encoder_embedding = embed_layer(encoder_inputs)
    encoder_LSTM = LSTM(HIDDEN_DIM, return_state=True)
    encoder_outputs, state_h, state_c = encoder_LSTM(encoder_embedding)
    encoder_states = [state_h, state_c]
```
- Another RNN acts as "decoder", takes in output of encoder and returns next characters of target sequence, given previous character of the target sequence.
```
    decoder_inputs = Input(shape=(MAX_LEN, ))
    decoder_embedding = embed_layer(decoder_inputs)
    decoder_LSTM = LSTM(HIDDEN_DIM, return_state=True, return_sequences=True)
    decoder_outputs, _, _ = decoder_LSTM(
        decoder_embedding, initial_state=encoder_states)
    outputs = TimeDistributed(
        Dense(VOCAB_SIZE, activation='softmax'))(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], outputs)
```
- The model is trained to turn the target sequence into the same sequences but offset by one timestep in the future, this process is called *teacher forcing* in this context.

![Encoder-Decoder](https://www.oreilly.com/library/view/deep-learning-essentials/9781785880360/assets/41162b03-716e-4290-a974-4a390fb904fe.png)

## Inference Mode
In order to decode an unknown sequence a different approach is taken:
- Encode the input sequence into state vectors
- Start with a dummy sequence with start token(here *bos*)
- Feed state vectors and dummy sequence to the decoder
- Append sampled word to the target sequence
- Repeat until sapled character is end-of-sequence token(here *eos*) or limit is reached
```
    encoder_model = Model(encoder_inputs, encoder_states)
    decoder_state_input_h = Input(shape=(None,))
    decoder_state_input_c = Input(shape=(None,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_LSTM(
    decoder_embedding, initial_state=decoder_states_inputs)

```
## Model Summary
### Data
- We'll be using [Cornell Movie--Dialogs Corpus](https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html) as our dataset
- Data is extracted out using `process_data.py` as: `encoders_inputs.txt`- context;`deocoder_inputs.txt`- response
- `<BOS>` and `<EOS>` tags are added to `decoder_inputs` to mark sequence beginning and end 
### Algorithm Summary
- Sentences are tokenized, indexed and [glove](https://nlp.stanford.edu/projects/glove/) embeddings are used; Shape(samples, max_limit, glove_dim)
- Embedding matrix; Shape(vocab, golve_dim)
#### Encoder
- Encoder input - Sequence; Shape(None, 20)
- Encoder output - States; Shape(None, 200)
#### Decoder
- Decoder input - States; Shape(None, 20)
- Decoder output - Outputs; Shape(None, 20, 200)
- Time distributed - Outputs; Shape(None, 20, 15000)

![Love](https://forthebadge.com/images/badges/built-with-love.svg)