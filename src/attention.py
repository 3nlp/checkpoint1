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
        #outputs=outputs[-1,:,:].unsqueeze(0)        
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
    def __init__(self, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(DecoderRNN, self).__init__()
        
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_size,hidden_size)
        
        self.wp1=  nn.Linear(hidden_size*2,hidden_size)
        self.wq=  nn.Linear(hidden_size*2,hidden_size)
        self.ws = nn.Linear(hidden_size,hidden_size)
        self.v =  nn.Linear(hidden_size,1)
        
        self.gru = nn.GRU(hidden_size*2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size*2, output_size)

        
    
    def forward(self, word_input, last_hidden, encoder_outputs,question_embedding,decode=False):
        
        if(decode):
            bs=1
        else:
            bs=batch_size
        

        word_embedded = self.embedding(word_input).unsqueeze(0).permute(1,0,2) # S=1 x B x N  

        seq_len=encoder_outputs.shape[0]
        
        question_embedding=self.wq(question_embedding)

        p_comp=self.wp1(encoder_outputs.contiguous().view(-1,encoder_outputs.size(2)))
        

        h_comp=self.ws(last_hidden.squeeze(0)).repeat(seq_len,1)
        q_comp=question_embedding.squeeze(0).repeat(seq_len,1)


        attn_weights=p_comp+h_comp+q_comp
        attn = self.v(attn_weights)
        attn = F.softmax(attn.view(bs,seq_len))
        
        context = torch.bmm(attn.unsqueeze(1),encoder_outputs.transpose(0, 1))
        context = context.transpose(0, 1)
        
        output, hidden = self.gru(context, last_hidden)
        
        p_vocab = F.softmax(self.out(torch.cat((word_embedded.squeeze(1), output.squeeze(0)), -1)))
        
        p_out=torch.log(p_vocab)  
        
        
        return p_out, hidden
            

def train(passage_input_batches, passage_input_lengths,question_input_batches,question_input_lengths, target_batches,target_lengths, question_encoder,passage_encoder, decoder, question_encoder_optimizer,passage_encoder_optimizer, decoder_optimizer, criterion):
    
    loss = 0
    passage_embedding_batches, _ = passage_encoder(passage_input_batches, passage_input_lengths, None)
    question_embedding, question_hidden = question_encoder(question_input_batches, question_input_lengths, None)
    
    question_encoder_optimizer.zero_grad()
    passage_encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    
    decoder_input = Variable(torch.LongTensor([29997]*batch_size))
    decoder_input=decoder_input.cuda()
    decoder_hidden = question_hidden[:decoder.n_layers] 
    
    max_target_length = max(target_lengths)

    Decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))        

    decoder_input = decoder_input.cuda()
    Decoder_outputs = Decoder_outputs.cuda()

    input_words=[]
    output_words=[]    
    for t in range(max_target_length):
        decoder_output, decoder_hidden, =decoder(decoder_input, decoder_hidden, passage_embedding_batches,question_embedding,decode=False)

        Decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] 


        topv,topi=decoder_output.data[0,:].topk(1)
        

        ni=topi
        
        
        if(int(target_batches[t,0].data)!=0):
            input_words.append(inv_vocab[int(target_batches[t,0].data)])
        if(int(ni)!=0):
            output_words.append(inv_vocab[int(ni)])

    print(input_words)
    print(output_words)
    
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),
        target_batches.transpose(0, 1).contiguous(),
        target_lengths)


    loss.backward()

    question_encoder_optimizer.step()
    passage_encoder_optimizer.step()
    decoder_optimizer.step()
    
    #torch.save(question_encoder.state_dict(), 'question_encoder_submission.pt')
    #torch.save(passage_encoder.state_dict(), 'passage_encoder_submission.pt')
    #torch.save(decoder.state_dict(), 'decoder_submission.pt')

    return loss.data[0]
    
# Pad a with the PAD symbol
def pad_seq(seq, max_length):
    seq += [0 for i in range(max_length - len(seq))]
    return seq


    
import matplotlib.pyplot as plt

import random    
def select_batch(batch_size):
    passage_seqs = []
    question_seqs=[]
    target_seqs = []
    i=0
    while(i <batch_size):
        index = random.randint(0,len(samples)-1)
        pair = samples[index]

        tar=pair[3]
        if(tar!=[]):
            i=i+1
            passage_seqs.append(pair[0])
            question_seqs.append(pair[1])
            target_seqs.append(pair[2])
            


    seq_pairs = sorted(zip(passage_seqs, question_seqs,target_seqs), key=lambda p: len(p[0]), reverse=True)
    passage_seqs,question_seqs,target_seqs = zip(*seq_pairs)
    
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
        
    return passage_var, passage_lengths, question_var, question_lengths,target_var,target_lengths



hidden_size = 512
n_layers = 1
dropout = 0.1
batch_size = 10

# Configure training/optimization
clip = 50.0

learning_rate = 0.0005
decoder_learning_ratio = 5
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


question_encoder_optimizer = optim.Adam(question_encoder.parameters(), lr=learning_rate)
passage_encoder_optimizer = optim.Adam(passage_encoder.parameters(), lr=learning_rate)

decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
criterion = nn.CrossEntropyLoss()

# Move models to GPU
question_encoder.cuda()
passage_encoder.cuda()
decoder.cuda()


plot_losses = []
print_loss_total = 0 
plot_loss_total = 0 


ecs = []
dcs = []
eca = 0
dca = 0

while epoch < n_epochs:
    epoch += 1
    
    passage_batches, passage_lengths, question_batches, question_lengths,target_batches,target_lengths = select_batch(batch_size)

    # Run the train function
    loss = train(passage_batches, passage_lengths,question_batches,question_lengths, target_batches,target_lengths, question_encoder,passage_encoder, decoder, question_encoder_optimizer,passage_encoder_optimizer, decoder_optimizer, criterion)

    print_loss_total += loss

    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary =  str(epoch / n_epochs * 100)+'% complete : loss=' + str(print_loss_avg)
        print(print_summary)

    