import os
import re
import sys
import json
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
from pprint import pprint
from collections import Counter
from nltk.corpus import stopwords
from scipy.sparse import coo_matrix
import re

from nltk.stem import PorterStemmer
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.wordnet import WordNetLemmatizer

tokenizer = WordPunctTokenizer()
stemmer = PorterStemmer()
lmtzr = WordNetLemmatizer()
#from tensorflow.contrib import learn

logging.getLogger().setLevel(logging.INFO)

def clean_str(s):
	s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
	s = re.sub(r" : ", ":", s)
	s = re.sub(r"\'s", " \'s", s)
	s = re.sub(r"\'ve", " \'ve", s)
	s = re.sub(r"n\'t", " n\'t", s)
	s = re.sub(r"\'re", " \'re", s)
	s = re.sub(r"\'d", " \'d", s)
	s = re.sub(r"\'ll", " \'ll", s)
	s = re.sub(r",", " , ", s)
	s = re.sub(r"!", " ! ", s)
	s = re.sub(r"\(", " \( ", s)
	s = re.sub(r"\)", " \) ", s)
	s = re.sub(r"\?", " \? ", s)
	s = re.sub(r"\s{2,}", " ", s)
	return s.strip().lower()

def load_embeddings(vocabulary):
	word_embeddings = {}
	for word in vocabulary:
		word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
	return word_embeddings


def pre_process(strng, enable_trivial=False):
    strng = clean_str(strng)
    words = []
    for i in strng.lower().split():
        i = lmtzr.lemmatize(i)
        i = str(i)
        words.append(i)
    strng = words
    return strng

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = pre_process(text)
    return tokens


from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer(analyzer='word', lowercase = True,tokenizer=tokenize_and_stem,
                     token_pattern = r'\b[a-zA-Z]+\b',ngram_range=(1,2),norm='l2')



def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
	"""Pad setences during training or prediction"""
	if forced_sequence_length is None: # Train
		sequence_length = max(len(x) for x in sentences)
	else: # Prediction
		logging.critical('This is prediction, reading the trained sequence length')
		sequence_length = forced_sequence_length
	logging.critical('The maximum length is {}'.format(sequence_length))

	padded_sentences = []
	for i in range(len(sentences)):
		sentence = sentences[i]
		num_padding = sequence_length - len(sentence)

		if num_padding < 0: # Prediction: cut off the sentence if it is longer than the sequence length
			logging.info('This sentence has to be cut off because it is longer than trained sequence length')
			padded_sentence = sentence[0:sequence_length]
		else:
			padded_sentence = sentence + [padding_word] * num_padding
		padded_sentences.append(padded_sentence)
	return padded_sentences

def build_vocab(sentences):
	word_counts = Counter(itertools.chain(*sentences))
	vocabulary_inv = [word[0] for word in word_counts.most_common()]
	vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
	return vocabulary, vocabulary_inv

def batch_iter(data, batch_size, num_epochs, shuffle=False):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int(data_size / batch_size) + 1

	for epoch in range(num_epochs):
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data

		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

def load_embedding_vectors_word2vec(vocabulary, filename, binary):
    # load embedding_vectors from the word2vec
    encoding = 'utf-8'
    with open(filename, "rb") as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        # initial matrix with random uniform
        embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
        if binary:
            binary_len = np.dtype('float32').itemsize * vector_size
            for line_no in range(vocab_size):
                word = []
                while True:
                    ch = f.read(1)
                    if ch == b' ':
                        break
                    if ch == b'':
                        raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                    if ch != b'\n':
                        word.append(ch)
                word = str(b''.join(word), encoding=encoding, errors='strict')
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                else:
                    f.seek(binary_len, 1)
        else:
            for line_no in range(vocab_size):
                line = f.readline()
                if line == b'':
                    raise EOFError("unexpected end of input; is count incorrect or file otherwise damaged?")
                parts = str(line.rstrip(), encoding=encoding, errors='strict').split(" ")
                if len(parts) != vector_size + 1:
                    raise ValueError("invalid vector on line %s (is this really the text format?)" % (line_no))
                word, vector = parts[0], list(map('float32', parts[1:]))
                idx = vocabulary.get(word)
                if idx != 0:
                    embedding_vectors[idx] = vector
        f.close()
        return embedding_vectors


def load_embedding_vectors_glove(vocabulary, filename, vector_size):
    # load embedding_vectors from the glove
    # initial matrix with random uniform
    embedding_vectors = np.random.uniform(-0.25, 0.25, (len(vocabulary), vector_size))
    f = open(filename)
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype="float32")
        idx = vocabulary.get(word)
        if idx != 0:
            embedding_vectors[idx] = vector
    f.close()
    return embedding_vectors

def load_data(filename):
	df = pd.read_csv(filename)
	selected = ['Category', 'Descript']
	non_selected = list(set(df.columns) - set(selected))

	df = df.drop(non_selected, axis=1)
	df = df.dropna(axis=0, how='any', subset=selected)
	df = df.reindex(np.random.permutation(df.index))

	labels = sorted(list(set(df[selected[0]].tolist())))
	num_labels = len(labels)
	one_hot = np.zeros((num_labels, num_labels), int)
	np.fill_diagonal(one_hot, 1)
	label_dict = dict(zip(labels, one_hot))

	x_raw= df[selected[1]].apply(lambda x: clean_str(x).split(' ')).tolist()
	y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

	x_raw = pad_sentences(x_raw)
	vocabulary, vocabulary_inv = build_vocab(x_raw)

	x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
	y = np.array(y_raw)
	return x, y, vocabulary, vocabulary_inv, df, labels

if __name__ == "__main__":
	train_file = './data/train_new.csv'
	load_data(train_file)
