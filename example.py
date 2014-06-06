#!/usr/bin/env python
#coding:utf-8

from hmm import Model
from preprocess import Process

    
train_dir = '/home/awen/code/hmm/hmm_tokenize/data/icwb2-data/training/pku_training.utf8'

'''
The number of the hidden states
B:a word at the start
E:a word at the end
M:a word at the middle
S:a word construct the tokenize
'''

S = ['B','E','M','S']
pro = Process(train_dir,S)
hidden_states,train=pro._statics_hidden()
word_count = pro._word_count(train)

observation = word_count.keys()

'''
The conf_prob is the probability of a observation in condition of a hidden state
The trans_prob is the probability of a hidden state trans to another
'''
conf_prob,trans_prob=pro._tran_conf_prob(train,word_count,hidden_states)

test = ["中华人民共和国今天成立了中国人民从此站起来了",
        "江泽民的三个代表是中国在社会主义改革过程中的智慧结晶"]
observations = pro._str2words(test)

phi = {'B':0.5,'E':0,'M':0,'S':0.5}
model = Model(S,observation,phi,trans_prob,conf_prob)
o_hstate = []
for obser in observations:
    o_hstate.append(model.decode(obser))

word_sequence = pro._word_sequence(observations,o_hstate)


     
