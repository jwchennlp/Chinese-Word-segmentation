#!/usr/bin/env python
#coding:utf-8

from hmm import Model
from preprocess import Process
import codecs
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


    
train_dir = '/home/awen/code/hmm/hmm_tokenize/data/icwb2-data/training/pku_training.utf8'
test_dir = '/home/awen/code/hmm/hmm_tokenize/data/icwb2-data/testing/pku_test.utf8'

'''
The number of the hidden states
B:a word at the start
E:a word at the end
M:a word at the middle
S:a word construct the tokenize
'''

S = ['B','E','M','S']
pro = Process(train_dir,S)
hidden_states,train=pro._statics()

pro_test = Process(test_dir,S)
userless,test = pro_test._statics()

test_wordcount = pro_test._word_count(test)

word_count = pro._word_count(train)

observation = word_count.keys()


'''
The conf_prob is the probability of a observation in condition of a hidden state
The trans_prob is the probability of a hidden state trans to another
This time add the smoothing method.
1.add 1 mehtod
'''
conf_prob,trans_prob=pro._tran_conf_prob(train,test_wordcount,word_count,hidden_states)

'''
test = [["中华人民共和国今天成立了中国人民从此站起来了"],
        ["江泽民的三个代表是中国在社会主义改革过程中的智慧结晶"],
        ["人民日报称改革开发的伟大旗帜要坚定不移动的走下去"],
        ["日理万机的周总理"],
        ["国务院今天颁发了关于农业的改革方向前进步伐"],
        ["机器学习及其翻译激起了人们极其浓厚的兴趣"],
        ["中共中央书记"]]
observations = pro._str2words(test)
'''
observations = test

phi = {'B':0.5,'E':0,'M':0,'S':0.5}
model = Model(S,observation,phi,trans_prob,conf_prob)
o_hstate = []

for obser in observations:
    '''
    Notice,if a setence is too long,when we use viterbi algorithm it may result in the beta = 0
    There are two solution,one is split the setence into serval sub_setence,another is use log function for the viterbi 
    here we select the first method
    '''
    length = len(obser)
    index,sub_obser,state= 0,[],[]
    while index < length:
        sub_obser.append(obser[index])
        if obser[index] == '。' or obser[index]=='，':
            sub_state = model.decode(sub_obser)
            sub_obser = []
            state += sub_state
        elif index == length-1:
            sub_state = model.decode(sub_obser)
            sub_obser = []
            state += sub_state
        index += 1
    o_hstate.append(state)

word_sequence = pro._word_sequence(observations,o_hstate)
print word_sequence[3]


     
