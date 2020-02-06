import os
import re
import codecs
import string
import pickle
import operator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import config as c


def list_to_txt(inputs, file_path):
    with open(file_path, 'w') as f:
        for input in inputs:
            f.write(input + '\n')


def clean_text(text):
    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", text)

    return text


def process_data(data_path):
    with codecs.open(c.data_path, "rb", encoding="utf-8", errors="ignore") as f:
        lines = f.read().split("\n")
        conversations = []
        for line in lines:
            data = line.split(" +++$+++ ")
            conversations.append(data)

    chats = {}
    for tokens in conversations:
        if len(tokens) > 4:
            idx = tokens[0][1:]
            chat = tokens[4]
            chats[int(idx)] = chat

    sorted_chats = sorted(chats.items(), key=lambda x: x[0])
    sorted_chats

    conves_dict = {}
    counter = 1
    conves_ids = []

    for i in range(1, len(sorted_chats)):
        if (sorted_chats[i][0] - sorted_chats[i-1][0]) == 1:
            conves_ids.append(sorted_chats[i-1][1])
            if sorted_chats[i-1][1] not in conves_ids:
                conves_ids.append(sorted_chats[i-1][1])
        conves_ids.append(sorted_chats[i][1])
        if (sorted_chats[i][0] - sorted_chats[i-1][0]) > 1:
            conves_dict[counter] = conves_ids
            conves_ids = []
        counter += 1

    context_and_target = []
    for conves in conves_dict.values():
        if len(conves) % 2 != 0:
            conves = conves[:-1]
        for i in range(0, len(conves), 2):
            context_and_target.append((conves[i], conves[i+1]))

    context, target = zip(*context_and_target)
    context = list(context)
    target = list(target)
    context = [clean_text(x) for x in context]
    bos = "<BOS>"
    eos = "<EOS>"
    target = [bos + clean_text(x) + eos for x in target]
    list_to_txt(context, c.encoder_input)
    list_to_txt(target, c.decoder_input)


if __name__ == '__main__':
    process_data(c.data_path)
