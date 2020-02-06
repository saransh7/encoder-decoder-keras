import os

# paths
project_dir = r'C:/Users/sbhatnagar4/Desktop/encoder-decoder-keras'
data_dir =  os.path.join(project_dir, r'data')
data_file = r'movie_lines.txt'
data_path = os.path.join(data_dir, data_file)

encoder_input = os.path.join(data_dir, r'encoder_inputs.txt')
decoder_input = os.path.join(data_dir, r'decoder_inputs.txt')

glove_file = r'glove.6B.50d.txt'
glove_path = os.path.join(data_dir, glove_file) 

# data params
VOCAB_SIZE = 15000 # max num of words in corpus
MAX_LEN = 20 # max num of tokens
SAMPLE_LEN = 10000

# model params
embedding_dim = 50 # from glove
hidden_dim = 300 # lstm hidden dim
epochs = 10
batch_size = 150
validation_split = 0.2

plot_loss = True