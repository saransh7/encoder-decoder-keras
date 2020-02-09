# encoder-decoder-keras
A keras implementaion of encoder decoder architecture for ChatBot

Demonstrates how to implement a basic word-level *sequence-to-sequence* model.

In the general case, input sequences and output sequences have different lengths (e.g. machine translation) and the entire input sequence is required in order to start predicting the target. This requires a more advanced setup, which is what people commonly refer to when mentioning "sequence to sequence models" with no further context.

## Training Mode
- A RNN acts as "encoder", processes input sequence and return its own internal state
- Another RNN acts as "decoder", takes in output of encoder and returns next characters of target sequence, given previous character of the target sequence.
- The model is trained to turn the target sequence into the same sequences but offset by one timestep in the future, this process is called *teacher forcing* in this context.

![Encoder-Decoder](https://www.oreilly.com/library/view/deep-learning-essentials/9781785880360/assets/41162b03-716e-4290-a974-4a390fb904fe.png)

## Inference Mode
In order to decode an unknown sequence a different approach is taken:
- Encode the input sequence into state vectors
- Start with a dummy sequence with start token(here *bos*)
- Feed state vectors and dummy sequence to the decoder
- Append sampled word to the target sequence
- Repeat until sapled character is end-of-sequence token(here *eos*) or limit is reached

![Love](https://forthebadge.com/images/badges/built-with-love.svg)