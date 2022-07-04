from datasets import load_dataset
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats
import torch.nn as nn
from datetime import datetime


loss_per_epoch = []

def get_data():
    train_data = load_dataset('sick', split = 'train')
    validation_data = load_dataset('sick', split = 'validation')
    test_data = load_dataset('sick', split = 'test')
    return train_data, validation_data, test_data

def process_data(data):
    sentence_a = data['sentence_A']
    sentence_b = data['sentence_B']
    
    scores = data['relatedness_score']
    sentence_scores = []

    for sent_a, sent_b, score in zip(sentence_a, sentence_b, scores):
        sentence = sent_a + ' [SEP] ' + sent_b
        sentence_scores.append([sentence, float(score/5)])
    return sentence_scores

class BertLinear(nn.Module):
    def __init__(self):
      super(BertLinear, self).__init__()
      self.bert = AutoModel.from_pretrained('bert-base-uncased')
      self.linear_layer = nn.Linear(768, 1, bias = True)
    
    def forward(self, **data):
        x = self.bert(**data).last_hidden_state[:, 0, :]
        x = self.linear_layer(x)
        return x

def train(model, tokenizer, optimizer, epochs, train_data_loader, validation_data_loader, device):
  max_loss = 10000000
  for epoch in range(epochs):
    running_loss = 0
    total_sentence = 0
    model.train()
    for i, data in enumerate(train_data_loader):
      sentence = list(data[0])
      scores = data[1]
      encoded_sentence = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
      encoded_sentence = encoded_sentence.to(device)
      embeddings_to_score = model(**encoded_sentence)
      normalization_score = F.sigmoid(embeddings_to_score)[:,0]
      true_scores = torch.tensor(scores, dtype = torch.float32)
      true_scores = true_scores.to(device)
      loss = F.mse_loss(normalization_score, true_scores)
      running_loss = running_loss + (loss.item() * len(sentence))
      total_sentence = total_sentence + len(sentence)
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      if i%10 == 0:
        sentence_to_print = "Epoch : "+str(epoch+1)+", Iteration : "+str(i+1)+" , Loss : "+ str(loss.item())
        print(sentence_to_print)
    epoch_loss = running_loss / total_sentence
    loss_per_epoch.append(epoch_loss)
    sentence_to_print = "Epoch : "+str(epoch+1)+" , Loss : "+ str(epoch_loss)
    print(sentence_to_print)
    validation_loss = validation(model, tokenizer, epoch, validation_data_loader, device)
    if validation_loss < max_loss:
      max_loss = validation_loss
      model_dict = model.state_dict()
      torch.save(model_dict, 'bert_linear_model.pth')
    
  return loss_per_epoch


def validation(model, tokenizer, epoch, validation_data_loader, device):
  running_loss = 0
  total_sentence = 0
  model.eval()
  for i, data in enumerate(validation_data_loader):
    sentence = list(data[0])
    scores = data[1]
    encoded_sentence = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    encoded_sentence = encoded_sentence.to(device)
    embeddings_to_score = model(**encoded_sentence)
    normalization_score = F.sigmoid(embeddings_to_score)[:,0]
    true_scores = torch.tensor(scores, dtype = torch.float32)
    true_scores = true_scores.to(device)
    loss = F.mse_loss(normalization_score, true_scores)
    running_loss = running_loss + (loss.item() * len(sentence))
    total_sentence = total_sentence + len(sentence)
    if i%10 == 0:
      sentence_to_print = "Epoch : "+str(epoch+1)+", Validation Iteration : "+str(i+1)+" , Loss : "+ str(loss.item())
      print(sentence_to_print)
  validation_loss = running_loss / total_sentence
  sentence_to_print = "Epoch : "+str(epoch+1)+", Validation Loss : "+ str(validation_loss)
  print(sentence_to_print)
  return validation_loss
  
def test(model, tokenizer, test_data_loader, device):
  true_scores = []
  predicted_scores = []
  model.eval()
  ms_error = 0
  total = 0
  for i, data in enumerate(test_data_loader):
    sentence = list(data[0])
    scores = data[1]
    encoded_sentence = tokenizer(sentence, padding=True, truncation=True, return_tensors='pt')
    encoded_sentence = encoded_sentence.to(device)
    embeddings_to_score = model(**encoded_sentence)
    normalization_score = F.sigmoid(embeddings_to_score)[:,0]
    normalization_score = normalization_score.to('cpu')
    predicted_scores.extend(normalization_score.tolist())
    true_scores.extend(scores.tolist())
    total = total + len(sentence)
  s = torch.tensor(true_scores, dtype = torch.float32)
  ns = torch.tensor(predicted_scores, dtype = torch.float32, requires_grad = True)
  mse = F.mse_loss(ns, s)
  sp_cor, _ = stats.spearmanr(predicted_scores,true_scores)
  pr_cor, _ = stats.pearsonr(predicted_scores,true_scores)
  print('Spearman Correlation : ' + str(sp_cor))
  print('Pearson Correlation : ' + str(pr_cor))
  print('Mean Squared Error : ' + str(mse.item()))    

def load_models(device):
  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
  model = BertLinear()
  model.load_state_dict(torch.load('bert_linear_model.pth'))
  model.to(device)
  return model, tokenizer

def generate_loss_plot(epochs):
  import matplotlib.pyplot as plt
  epochs_ = list(range(1,epochs+1))
  plt.plot(epochs_, loss_per_epoch)
  plt.xlabel('Epoch')
  plt.ylabel('Loss') 
  plt.savefig('loss_curve', bbox_inches = "tight")
  plt.show()