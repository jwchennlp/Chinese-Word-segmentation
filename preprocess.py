#!/usr/bin/env python
#coding:utf-8

import codecs,re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')



class Process(object):
    def __init__(self,file_dir,S):
        self._file_dir = file_dir
        self._S = S
    
    '''
    def _read(self):
        f = open(self._file_dir,'rb')
        regex=re.compile("(?x) ( [\w-]+ | [\x80-\xff]{3} )")
        train = []
        for line in f.readlines():
            line = line.replace(' ','')
            line = line.replace('\r\n','')
            words = [w for w in regex.split(line) if w]
            if len(words) != 0:
                train.append(words)
        return train
    '''

    def _str2words(self,test):
        words =[]
        x=codecs.lookup("utf-8")
        for string in test:
            word = x.decode(string[0])[0]
            words.append(word)
        return words

    def _statics(self):
        f = codecs.open(self._file_dir,'rb',encoding = 'utf-8')
        hidden_states,train = [],[]
        for line in f.readlines():
            '''
            First make tag for the tokenize in the corpus
            '''
            hidden_state = ''
            words = []
            tokenizes = line.split()
            for token in tokenizes:
                length = len(token)
                if length == 1:
                    hidden_state += 'S'
                elif length==2:
                    hidden_state += 'BE'
                else:
                    hidden_state += 'B'+(length-2)*'M'+'E'
            '''
            Secong we should extart single word from the corpus
            '''
            line = line.replace(' ','')
            line = line.replace('\r\n','')
            for word in line:
                words.append(word)
            if len(words) >0:
                train.append(words)
                hidden_states.append(hidden_state)
        return (hidden_states,train)
            
    def _statics_hidden(self):
        '''
        First,get the tokenize result of the corpus,
        statics the hidden state of each word
        '''
        f = open(self._file_dir,'rb')
        hidden_states,train = [],[]
        regex=re.compile("(?x) ( [\w-]+ | [\x80-\xff]{3} )")
        for line in f.readlines():
            hidden_state = ''
            words = []
            tokenizes = line.split()
            for token in tokenizes:
                temp = [w for w in regex.split(token) if w]
                for t in temp:
                    words.append(t)
                length = len(temp)
                if length == 1:
                    hidden_state += 'S'
                elif length==2:
                    hidden_state += 'BE'
                else:
                    hidden_state += 'B'+(length-2)*'M'+'E'
            if len(words) != 0:
                train.append(words)
                hidden_states.append(hidden_state)
        return (hidden_states,train)

            
    def _word_count(self,train):
        word_count = {}
        for words in train:
            for word in words:
                if  word_count.has_key(word):
                    word_count[word] += 1
                else:
                    word_count[word] = 1
        return word_count
    
    def _convert(self,hidden_states):
        temp = []
        for index in range(len(hidden_states)):
            regex = re.compile("(\w{1})")
            states = [w for w in regex.split(hidden_states[index]) if w]
            if len(states) !=0:
                temp.append(states)
        return temp
    
    def _cal_trans(self,h_s):
        trans_prob,state_count = {},{}
        #intial
        for state in self._S:
            trans_prob[state]={}
            state_count[state] = 0
            for state_i in self._S:
                trans_prob[state][state_i]=0
        for i in range(len(h_s)):
            length = len(h_s[i])
            for j in range(length-1):
                s_from = h_s[i][j]
                s_to = h_s[i][j+1]
                trans_prob[s_from][s_to] += 1
                state_count[s_from] += 1
            state_count[h_s[i][length-1]] += 1
        print state_count
        for i in self._S:
            for j in self._S:
                trans_prob[i][j] /= float(state_count[i])
        return (trans_prob,state_count)
    
    def _cal_conf(self,h_s,test_wordcount,word_count,train,state_count):
        conf_prob = {}
        words = list(set(word_count.keys())|set(test_wordcount.keys()))
        print('The corpus has %d word'%(len(words)))
        for state in self._S:
            conf_prob[state] = {}
            for word in words:
                conf_prob[state][word] = 1
        for i in range(len(h_s)):
            length = len(h_s[i])
            for j in range(length):
                obser = train[i][j]
                hidden = h_s[i][j]
                conf_prob[hidden][obser] += 1
        for state in self._S:
            for word in words:
                if conf_prob[state][word] == 0:
                    continue
                else:
                    conf_prob[state][word] /= float(state_count[state])
        return conf_prob
        

    def _tran_conf_prob(self,train,test_wordcount,word_count,hidden_states):
        #convert the hidden_state string to list
        hidden_states = self._convert(hidden_states)
        trans_prob,state_count = self._cal_trans(hidden_states)
        conf_prob = self._cal_conf(hidden_states,test_wordcount,word_count,train,state_count)
        
        return (conf_prob,trans_prob)
        
    def _word_sequence(self,test,o_hstate):
        sequence = []
        f= open('./data/result','wb')
        for i in range(len(test)):
            if o_hstate[i][-1] == 'M':
                o_hstate[i][-1] = 'E'
            elif o_hstate[i][-1] == 'B':
                o_hstate[i][-1] = 'S'
            length = len(test[i])
            temp = []
            k = 0
            while k <length:
                if o_hstate[i][k]=='S':
                    temp.append(test[i][k])
                else :
                    s=test[i][k]
                    k+=1
                    while o_hstate[i][k] != 'E' :
                        s += test[i][k]
                        k +=1
                    s += test[i][k]
                    temp.append(s)
                k += 1
            f.write('%s\n'%(' '.join(temp)))
            sequence.append(' '.join(temp))
        f.close()
            
        return sequence
    
        


