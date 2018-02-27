from __future__ import absolute_import
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import pickle
import numpy as np
from masked_cross_entropy import *
from evaluation import evaluate

samples=pickle.load(open('samples_synth.pkl','rb'))

Vocab_dict=pickle.load(open('Vocab_synth.pkl','rb'))


l_max=100

inv_vocab = {v:k for k, v in Vocab_dict.items()}

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,n_layers=2, dropout=0.1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = nn.Embedding(input_size, hidden_size)     
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

        
    def forward(self, input_seqs, input_lengths, hidden=None):

        embedded = self.embedding(input_seqs)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)
        outputs=outputs[-1,:,:].unsqueeze(0)        
        return outputs, hidden
        
class qEncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size,n_layers=2, dropout=0.1):
        super(qEncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, hidden_size)

        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True)

        
    def forward(self, input_seqs, input_lengths, hidden=None):
        
        embedded = self.embedding(input_seqs)
        outputs, hidden = self.gru(embedded, hidden)
        outputs=outputs[-1,:,:].unsqueeze(0)
        return outputs, hidden


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, n_layers=2, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_size,hidden_size)

        self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size*2, output_size)

        
    
    def forward(self, word_input, last_hidden, encoder_output,decode=False):
        

        word_embedded = self.embedding(word_input).unsqueeze(0).permute(1,0,2)

        output, hidden = self.gru(encoder_output, last_hidden)


        p_vocab = F.softmax(self.out(torch.cat((word_embedded.squeeze(1), output.squeeze(0)), -1)))
        
        p_out=torch.log(p_vocab)  
        
        
        return p_out, hidden
        
hidden_size = 512
n_layers = 1
dropout = 0.1
batch_size = 1

clip = 50.0

learning_rate = 0.001
decoder_learning_ratio = 1
n_epochs = 50000
epoch = 0
plot_every = 10000
print_every = 1
evaluate_every = 10
Vocab_size=len(Vocab_dict.keys())

# Initialize models
question_encoder=qEncoderRNN(Vocab_size, hidden_size,n_layers=1, dropout=0.1)
passage_encoder=EncoderRNN(Vocab_size, hidden_size,n_layers=1, dropout=0.1)

decoder = DecoderRNN(hidden_size, Vocab_size, dropout_p=0.1)

question_encoder.cuda()
passage_encoder.cuda()
decoder.cuda()


question_encoder.load_state_dict(torch.load('question_encoder_basic.pt'))
passage_encoder.load_state_dict(torch.load('passage_encoder_basic.pt'))
decoder.load_state_dict(torch.load('decoder_basic.pt'))



def test(passage_input_batches, passage_input_lengths,question_input_batches,question_input_lengths, target_batches,target_lengths, question_encoder,passage_encoder, decoder):
 

    passage_embedding_batches, _ = passage_encoder(passage_input_batches, passage_input_lengths, None)
    question_embedding, question_hidden = question_encoder(question_input_batches, question_input_lengths, None)

    decoder_input = Variable(torch.LongTensor([29997]*batch_size))
    decoder_input=decoder_input.cuda()
    decoder_hidden = question_hidden[:decoder.n_layers] 
    
    max_target_length = max(target_lengths)

    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))        
    decoder_input = decoder_input.cuda()
    all_decoder_outputs = all_decoder_outputs.cuda()

    input_words=[]
    output_words=[]    
    for t in range(max_target_length):
        decoder_output, decoder_hidden, =decoder(decoder_input, decoder_hidden, passage_embedding_batches,decode=False)

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] 


        topv,topi=decoder_output.data[0,:].topk(1)
        

        ni=topi
        
        
        if(int(target_batches[t,0].data)!=0):
            input_words.append(inv_vocab[int(target_batches[t,0].data)])
        if(int(ni)!=0):
            output_words.append(inv_vocab[int(ni)])

    reference = ' '.join(input_words)
    hypothesis = ' '.join(output_words)
    
    print('#')
    print(input_words)
    print(output_words)
    return reference,hypothesis


    
    
def pad_seq(seq, max_length):
    seq += [0 for i in range(max_length - len(seq))]
    return seq


mean=0
bleu=0
rogue=0
for index in range(len(samples)):
    pair = samples[index]

    if(pair[2]==[]):
        continue
    passage_seqs = [pair[0]]
    question_seqs = [pair[1]]
    target_seqs = [pair[2]]
            
            
    passage_lengths = [len(s) for s in passage_seqs]
    passage_padded = [pad_seq(s, max(passage_lengths)) for s in passage_seqs]
    question_lengths = [len(s) for s in question_seqs]
    question_padded = [pad_seq(s, max(question_lengths)) for s in question_seqs]
    target_lengths = [len(s) for s in target_seqs]
    target_padded = [pad_seq(s, max(target_lengths)) for s in target_seqs]


    passage_var = Variable(torch.LongTensor(passage_padded)).transpose(0, 1)
    question_var = Variable(torch.LongTensor(question_padded)).transpose(0, 1)
    target_var = Variable(torch.LongTensor(target_padded)).transpose(0, 1)


    passage_var = passage_var.cuda()
    question_var = question_var.cuda()
    target_var=target_var.cuda()
    
    reference,hypothesis=test(passage_var, passage_lengths,question_var,question_lengths, target_var,target_lengths, question_encoder,passage_encoder, decoder)
    print(evaluate.get_score(reference, hypothesis))
    bleu+=evaluate.get_score(reference, hypothesis)['Bleu_1']
    rogue+=evaluate.get_score(reference, hypothesis)['ROUGE_L']

print(bleu)
print(rogue)
print(bleu/len(samples))
print(rogue/len(samples))

