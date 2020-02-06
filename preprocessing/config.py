import os

project_dir = r'C:/Users/sbhatnagar4/Desktop/encoder-decoder-keras'
data_dir = r'C:/Users/sbhatnagar4/Desktop/encoder-decoder-keras/data'
data_file = r'movie_lines.txt'
data_path = os.path.join(data_dir, data_file)

encoder_input = os.path.join(data_dir, 'encoder_inputs.txt')
decoder_input = os.path.join(data_dir, 'decoder_inputs.txt')
