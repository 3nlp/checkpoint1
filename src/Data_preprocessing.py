from __future__ import absolute_import
from __future__ import print_function
import re
import json
import pickle
import operator
import numpy as np
from nltk.tokenize import word_tokenize, sent_tokenize

#Vocab_dict=pickle.load(open('Vocab_synth.pkl','rb'))

'''Data prep'''

def sublist_exists(sl, l):
    count=0

    n=len(sl)
    
    result=False

    for word in sl:
        if word in l:
            count=count+1

    if(count>0.7*n):
        result=True
        
    return result


Questions=[]
Passages_word=[]
Passages_sent=[]
Vocab_dict={}
Answers=[]
Is_selected=[]
word_count=1
Target=[]
how_many=0

def normalizeString(s):
    s.encode('ascii','ignore')
    s = re.sub(r"([?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z?]+", r" ", s)
    s=s.lower()
    return s
  
with open('dev.json') as f:
    line = f.readline()

    while(line):
        how_many=how_many+1
        print(how_many)
        dic = json.loads(line)
        question=normalizeString(dic['query'])
        question = word_tokenize(question)
        
        for word in question:
            if(word not in Vocab_dict):
                Vocab_dict[word]=1
            else:
                Vocab_dict[word]+=1
                
                
        answer = dic['answers']

        if answer == []:
            line = f.readline()
            continue
        else:
             answer1 = word_tokenize(normalizeString(answer[0]))
        
        for word in answer1:
            if(word not in Vocab_dict):
               Vocab_dict[word]=1
            else:
                Vocab_dict[word]+=1
        

        references=''
        passages=''
        passages_sent=[]
        is_selected=[]
        for i in dic['passages']:
            passage_text=normalizeString(i['passage_text'])
            
            
            if int(i['is_selected']) == 1:
                references+=passage_text
            is_selected.append(int(i['is_selected']))
            passages+=passage_text
            
        #print(passages)           
        passage_word=word_tokenize(passages)
        #print(passage_word)

        passage_sent = sent_tokenize(passages)

        

        passage_sent = [word_tokenize(sent) for sent in passage_sent]

        '''
        for word in passage_word:
            
            if(word not in Vocab_dict):
                Vocab_dict[word]=1
            else:
                Vocab_dict[word]+=1
        '''
        
        word_pos=0     
        for i in range(len(passage_sent)):
           sent=passage_sent[i]
           if(sublist_exists(answer1, sent)):
               sent_idx = i
               target=[word_pos, word_pos+len(sent)-1]
               word_pos=word_pos+len(sent)
               break
           else:
               word_pos=word_pos+len(sent)
               target=[]

             
        line = f.readline()

        #print(target)

        '''
        if(target!=[]):
            print('Target')
            print(passage_word[target[0]:target[1]])
        '''
        Passages_word.append(passage_word)
        Passages_sent.append(passage_sent)
        Answers.append(answer1)
        Questions.append(question)
        Is_selected.append(is_selected)
        Target.append(target)
        

sorted_x = sorted(Vocab_dict.items(), key=operator.itemgetter(1))[::-1]
Vocab_dict=dict(sorted_x[:29997])

keys=Vocab_dict.keys()
values=range(len(Vocab_dict.keys()))
Vocab_dict=dict(zip(keys,values))
Vocab_dict['<SOS>']=len(Vocab_dict.keys())

Vocab_dict['<EOS>']=len(Vocab_dict.keys())+1
Vocab_dict['UNK']=len(Vocab_dict.keys())+2


     
def word2index(words,dictionary):
    wordIndices = []
    for word in words:
        if(word in dictionary):
            wordIndices.append(dictionary[word])
        else:
            wordIndices.append(dictionary['UNK'])

    return wordIndices	

   
samples=[]    

for index in range(len(Passages_word)):
    passage=word2index(Passages_word[index],Vocab_dict)
    question=word2index(Questions[index],Vocab_dict)
    answer=word2index(Answers[index],Vocab_dict)
    #target=Target[index]
    tar=np.ones(len(passage))
    #tar[target[0]:target[1]+1]=np.ones(target[1]-target[0]+1)

    samples.append([passage,question,answer,tar])


with open('samples_dev.pkl', 'wb') as output:  
    pickle.dump(samples, output, pickle.HIGHEST_PROTOCOL)  

with open('Vocab.pkl', 'wb') as output:  
    pickle.dump(Vocab_dict, output, pickle.HIGHEST_PROTOCOL)