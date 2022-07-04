import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import similarity_score
import math
import numpy as np
import random



"""
Wrapper class using Pytorch nn.Module to create the architecture for our model
Architecture is based on the paper: 
A STRUCTURED SELF-ATTENTIVE SENTENCE EMBEDDING
https://arxiv.org/pdf/1703.03130.pdf

for SelfAttention model the following link was used for help
https://github.com/timbmg/Structured-Self-Attentive-Sentence-Embedding
"""


class SiameseBiLSTMAttention(nn.Module):
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
    ):
        super(SiameseBiLSTMAttention, self).__init__()
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
        
        lstm_output_sent1, (hidden, cell) = self.bilstm(embeddings_sent1)
        sent_1_attentions = self.attention(lstm_output_sent1)
        sent1_embeddings = torch.bmm(sent_1_attentions, lstm_output_sent1)
        sent1_embeddings =  sent1_embeddings.view(-1, self.att_output_dim * self.lstm_hidden_size * self.lstm_directions)
        output_sent1 = self.linear(sent1_embeddings)
        
        embeddings_sent2 = self.embeddings(sent2_batch)

        lstm_output_sent2, (hidden, cell) = self.bilstm(embeddings_sent2)
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