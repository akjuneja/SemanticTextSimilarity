import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import similarity_score
import math
import numpy as np
import random

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


"""
Wrapper class using Pytorch nn.Module to create the architecture for our model
Architecture is based on the paper: 
A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
https://arxiv.org/pdf/1703.03130.pdf

for MultiHeadAttention, Transformer, TransformerEncoder the following link was used for help
https://www.kaggle.com/arunmohan003/transformer-from-scratch-using-pytorch

for SelfAttention model the following link was used for help
https://github.com/timbmg/Structured-Self-Attentive-Sentence-Embedding
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, batch_size, seq_len, num_heads, emb_dim):
        super(MultiHeadAttention, self).__init__()
        
        self.head_dim = int(emb_dim / num_heads)
        self.query = nn.Linear(self.head_dim , self.head_dim ) 
        self.key = nn.Linear(self.head_dim  , self.head_dim)
        self.value = nn.Linear(self.head_dim , self.head_dim)
        self.final_output = nn.Linear(num_heads * self.head_dim, emb_dim) 
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.num_heads = num_heads

    def forward(self,k, q ,v):
        
        k = k.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim) 
        q = q.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim) 
        v = v.view(self.batch_size, self.seq_len, self.num_heads, self.head_dim) 
       
        k = self.key(k) 
        q = self.query(q)   
        v = self.value(v)

        q = q.transpose(1,2)  
        k = k.transpose(1,2) 
        v = v.transpose(1,2) 
       
        k = k.transpose(-1,-2) 
        mul = torch.matmul(q, k)
        mul = mul / math.sqrt(self.head_dim)
        scores = F.softmax(mul, dim=-1)
        scores = torch.matmul(scores, v)
        concat = scores.transpose(1,2).contiguous().view(self.batch_size, self.seq_len, self.head_dim*self.num_heads)
        output = self.final_output(concat)
        return output

class Transformer(nn.Module):
    def __init__(self, emb_dim, seq_len, num_heads, batch_size, expansion_factor):
        super(Transformer, self).__init__()
        
        self.attention = MultiHeadAttention(batch_size, seq_len, num_heads, emb_dim)
        
        self.linear_1 = nn.Linear(emb_dim, expansion_factor*emb_dim)
        self.linear_2 = nn.Linear(expansion_factor*emb_dim, emb_dim)
        
        self.relu_1 = nn.ReLU()
        
        self.norm_1 = nn.LayerNorm(emb_dim) 
        self.norm_2 = nn.LayerNorm(emb_dim)

        self.dropout_1 = nn.Dropout(0.2)
        self.dropout_2 = nn.Dropout(0.2)
        
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.batch_size = batch_size
        

    def forward(self,k,q,v):
        
        attention_out_1 = self.attention(k, q, v)  
        attention_out_2 = attention_out_1 + v
        norm_1_out_1 = self.norm_1(attention_out_2)
        dropout_1 = self.dropout_1(norm_1_out_1)

        linear_1_out = self.linear_1(dropout_1)
        relu_1 = self.relu_1(linear_1_out)
        linear_2_out = self.linear_2(relu_1)
    
        linear_2_final = linear_2_out + dropout_1
        norm_2_out_1 = self.norm_2(linear_2_final)
        dropout_2 = self.dropout_2(norm_2_out_1) 
       

        return dropout_2



class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, emb_dim, layers, batch_size, num_heads, expansion_factor=4):
        super(TransformerEncoder, self).__init__()

        self.layers = nn.ModuleList([Transformer(emb_dim, seq_len, num_heads, batch_size, expansion_factor) for i in range(layers)])
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x,x,x)

        return x

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import similarity_score
import math


"""
Wrapper class using Pytorch nn.Module to create the architecture for our model
Architecture is based on the paper: 
A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
https://arxiv.org/pdf/1703.03130.pdf
"""


class SiameseTransformerEncoderBiLSTMAttention(nn.Module):
    def __init__(
        self,
        batch_size,
        output_size,
        hidden_size,
        vocab_size,
        embedding_size,
        embedding_weights,
        lstm_layers,
        device,
        bidirectional,
        self_attention_config,
        fc_hidden_size,
        seq_len,
        num_heads,
        transformer_layers
    ):
        super(SiameseTransformerEncoderBiLSTMAttention, self).__init__()
        """
        Initializes model layers and loads pre-trained embeddings from task 1
        """
        ## model hyper parameters
        self.self_attention_config = self_attention_config
        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm_hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.lstm_layers = lstm_layers
        self.device = device
        self.bidirectional = bidirectional
        self.fc_hidden_size = fc_hidden_size
        self.lstm_directions = (
            2 if self.bidirectional else 1
        )  ## decide directions based on input flag
        
        pass
        ## model layers
        # TODO initialize the look-up table.
        self.embeddings = nn.Embedding(len(embedding_weights), embedding_size)
        self.transformer_encoder = TransformerEncoder(seq_len, embedding_size, transformer_layers, 
                                                      self.batch_size, num_heads)
        # TODO assign the look-up table to the pre-trained fasttext word embeddings.
        self.embeddings.weight = torch.nn.Parameter(embedding_weights)

        ## TODO initialize lstm layer
        self.bilstm = nn.LSTM(input_size = embedding_size, 
                              hidden_size = self.lstm_hidden_size, 
                              num_layers = self.lstm_layers,
                              bidirectional = self.bidirectional,
                              batch_first = self.bidirectional)

        

        ## TODO initialize self attention layers
        self.att_hidden_dim = self_attention_config["hidden_size"]
        self.att_output_dim = self_attention_config["output_size"]

        self.attention = SelfAttention(self.lstm_hidden_size * self.lstm_directions, 
                                    self.att_hidden_dim,
                                    self.att_output_dim)

        ## incase we are using bi-directional lstm we'd have to take care of bi-directional outputs in
        ## subsequent layers
        self.linear = nn.Linear(self.att_output_dim * self.lstm_hidden_size * self.lstm_directions, 
                                self.fc_hidden_size)


    def init_hidden(self, batch_size):
        """
        Initializes hidden and context weight matrix before each
                forward pass through LSTM
        """
        pass
        # TODO implement
        

    def forward_once(self, batch, lengths):
        """
        Performs the forward pass for each batch
        """

        ## batch shape: (batch_size, seq_len)
        ## embeddings shape: ( batch_size, seq_len, embedding_size)

        # TODO implement
        pass

    def forward(self, sent1_batch, sent2_batch):
        """
        Performs the forward pass for each batch
        """
        ## TODO init context and hidden weights for lstm cell
        pass
        # TODO implement forward pass on both sentences. calculate similarity using similarity_score()
        
        embeddings_sent1 = self.embeddings(sent1_batch)
        transformer_sent1 = self.transformer_encoder(embeddings_sent1)
        lstm_output_sent1, (hidden, cell) = self.bilstm(embeddings_sent1)
        sent_1_attentions = self.attention(lstm_output_sent1)
        sent1_embeddings = torch.bmm(sent_1_attentions, lstm_output_sent1)
        sent1_embeddings =  sent1_embeddings.view(-1, self.att_output_dim * self.lstm_hidden_size * self.lstm_directions)
        output_sent1 = self.linear(sent1_embeddings)
        
        embeddings_sent2 = self.embeddings(sent2_batch)
        transformer_sent2 = self.transformer_encoder(embeddings_sent2)
        lstm_output_sent2, (hidden, cell) = self.bilstm(transformer_sent2)
        sent_2_attentions = self.attention(lstm_output_sent2)
        sent2_embeddings = torch.bmm(sent_2_attentions, lstm_output_sent2)
        sent2_embeddings =  sent2_embeddings.view(-1, self.att_output_dim * self.lstm_hidden_size * self.lstm_directions)
        output_sent2 = self.linear(sent2_embeddings)
        
        similarity_scores = similarity_score(output_sent1, output_sent2)
        return similarity_scores, sent_1_attentions, sent_2_attentions
        


class SelfAttention(nn.Module):
    """
    Implementation of the attention block
    """

    def __init__(self, input_size, hidden_size, output_size):
        super(SelfAttention, self).__init__()
        # TODO implement
        
        self.linear_1 = nn.Linear(input_size, hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.linear_2 = nn.Linear(hidden_size, output_size, bias=False)

    ## the forward function would receive lstm's all hidden states as input
    def forward(self, attention_input):
        # TODO implement
        x1 = self.linear_1(attention_input)
        x2 = self.tanh(x1)
        x3 = self.linear_2(x2)

        weights = F.softmax(x3, dim=1)
        weights = weights.transpose(1, 2)

        return weights