import tensorflow as tf
tf.reset_default_graph()
import nltk as nltk
import seq2seq_wrapper
import numpy as np
import data as data
import data_utils
#import importlib
#importlib.reload(data)




def getModel():
    # load data from pickle and npy files
    metadata, idx_q, idx_a = data.load_data(PATH='datasets/cornell_corpus/')
    (trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_dataset(idx_q, idx_a)
    train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, 32)
    
    print len(trainX)
    test_batch_gen = data_utils.rand_batch_gen(testX, testY, 256)
    input_ = test_batch_gen.next()[0]
    xseq_len = 25
    yseq_len = 25
    batch_size = 16
    xvocab_size = len(metadata['idx2w'])
    yvocab_size = xvocab_size
    emb_dim = 1024
    model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/cornell_corpus/',
                               emb_dim=emb_dim,
                               num_layers=3
                               )
    sess = model.restore_last_session()
    output = model.predict(sess, input_[0:25])
    print(output)
    return model, sess



def chat(model,sess,question):
    # load data from pickle and npy files
    metadata, idx_q, idx_a = data.load_data(PATH='datasets/cornell_corpus/')

    w_2_id = dict(metadata['w2idx'])
    id_2_w = metadata['idx2w']

    questions = question.lower()
    qtokenized =  [w.strip() for w in questions.split(' ') if w]
    input_array = np.array([data.pad_seq(qtokenized, w_2_id, 25)])
    input_array = input_array.T
    print(input_array)
    emb_dim = 400

    batch_size = 16
    xvocab_size = len(metadata['idx2w'])
    yvocab_size = xvocab_size
    emb_dim = 1024

    predict = model.predict(sess=sess, X=input_array)

    print("predict",predict)
    response = [id_2_w[i] for i in predict[0]]
    print("response",response)


model, sess = getModel()
chat(model,sess,"What do you think about life? Is life fair for you? You are a jerk though!")
chat(model,sess, "well i really think hes got a chance")
chat(model,sess,"my god these people are insane")
chat(model,sess,"You never wanted to go out with 'me, did you?")
chat(model,sess,"How do you get your hair to look like that?")
chat(model,sess,"You think you ' re the only sophomore at the prom?")
