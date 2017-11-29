import numpy as np
import re
import itertools
from collections import Counter


def clean_str(string):
    
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data_and_labels(all_data_file):
    
    # Load data from files
    # What to do if the file is currently being edited by someone. A lock has been set on the file. So keep A backup that is swapped in and out.
    intent = []
    all_examples = list(open(all_data_file, "r").readlines())
    x_text = []
    for example in all_examples:

        splitIndex = example.index("-")
        current_intent = example[0:splitIndex]
        current_intent_index = intent.index(current_intent) if current_intent in intent else -1
        if current_intent_index == -1:
            intent.append(current_intent)
        x_text.append(example[splitIndex+1:].strip())
    number_of_intents = len(intent)
    one_hot_vector = np.ndarray(shape=(number_of_intents,number_of_intents))
    for encode_column in range(0,number_of_intents):
        #one_hot_vector_row = []
        for encode_row in range(0,number_of_intents):
            if encode_column == encode_row:
                one_hot_vector[encode_row][encode_column] = 1
            else:
                one_hot_vector[encode_row][encode_column] = 0
        #one_hot_vector.append(one_hot_vector_row)
    print(one_hot_vector)
    x_text = [clean_str(sent) for sent in x_text]
    y = np.ndarray(shape=(len(x_text),number_of_intents))
    index = 0
    for example in all_examples:
        splitIndex = example.index("-")
        current_intent = example[0:splitIndex]
        y[index] = one_hot_vector[intent.index(current_intent)]
        index = index + 1
    #print(type(y))
    #print(y)
   #print(y[0])
    #print(type(y[0]))
    #quit()
  
    if len(x_text) == len(y):
        return [x_text, y]
    else:
        print("The length of the training labels and the labels assigned to them do not match")
        quit();

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
