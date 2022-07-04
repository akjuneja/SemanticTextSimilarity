import torch
from torch import nn
import torch.nn.functional as F
import logging
from tqdm import tqdm
from torch.autograd import Variable
from scipy import stats
import numpy as np
import random


logging.basicConfig(level=logging.INFO)

"""
Script for training the neural network and saving the better models 
while monitoring a metric like accuracy etc
"""


def train_model(model, optimizer, dataloader, data, max_epochs, config_dict):
    device = config_dict["device"]
    criterion = nn.MSELoss()
    max_accuracy = 0.0
    min_loss = 100000000
    loss_per_epoch = []
    device = config_dict['device']
    model.to(device)
    for epoch in tqdm(range(max_epochs)):
        model.train()
        # TODO implement
        running_loss = 0
        total_sentence = 0
        predicted_scores = []
        true_targets = []
        for i, data in enumerate(dataloader['train']):
            sent_a,sent_b,_,_,targets,_,_ = data
            sent_a = sent_a.to(device)
            sent_b = sent_b.to(device)
            targets = targets.to(device)
            
            similarity_scores, sent_a_attentions, sent_b_attentions = model(sent_a, sent_b)

            pen_term_a = attention_penalty_loss(sent_a_attentions,
                                                        model.self_attention_config["penalty"] , 
                                                        device)
            
            pen_term_b = attention_penalty_loss(sent_b_attentions,
                                                        model.self_attention_config["penalty"] , 
                                                        device)


            loss = F.mse_loss(similarity_scores, targets) + pen_term_a + pen_term_b
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss = running_loss + (loss.item() * len(sent_a))
            total_sentence = total_sentence + len(sent_a)
            targets = targets.to('cpu')
            similarity_scores = similarity_scores.to('cpu')
            predicted_scores.extend(similarity_scores.tolist())
            true_targets.extend(targets.tolist())
            if i%10 == 0:
                sentence_to_print = "Epoch : "+str(epoch+1)+", Iteration : "+str(i+1)+" , Loss : "+ str(loss.item())
                print(sentence_to_print)
        epoch_loss = running_loss / total_sentence
        loss_per_epoch.append(epoch_loss)
        sentence_to_print = "Epoch : "+str(epoch+1)+" , Loss : "+ str(epoch_loss)
        print(sentence_to_print)
        s = torch.tensor(true_targets, dtype = torch.float32)
        ns = torch.tensor(predicted_scores, dtype = torch.float32, requires_grad = True)
        mse = F.mse_loss(ns, s)
        mes = mse.item()
        acc, _ = stats.spearmanr(predicted_scores,true_targets)
        pr_cor, _ = stats.pearsonr(predicted_scores,true_targets)


        
        # TODO: computing accuracy using sklearn's function
        

        ## compute model metrics on dev set
        val_acc, val_loss = evaluate_dev_set(
            model, data, criterion, dataloader, config_dict, device, epoch
        )

        
        if val_acc > max_accuracy:
            max_accuracy = val_acc
            logging.info(
                "new model saved"
            )  ## save the model if it is better than the prior best
            torch.save(model.state_dict(), "{}.pth".format(config_dict["model_name"]))

        logging.info(
            "Train loss: {} - acc: {} -- Validation loss: {} - acc: {}".format(
                epoch_loss, acc, val_loss, val_acc
            )
        )
    return max_accuracy


def evaluate_dev_set(model, data, criterion, dataloader, config_dict, device, epoch):
    """
    Evaluates the model performance on dev data
    """
    logging.info("Evaluating accuracy on dev set")
    running_loss = 0
    total_sentence = 0
    model.eval()
    predicted_scores = []
    true_targets = []
    device = config_dict['device']
    for i, data in enumerate(dataloader['validation']):
        sent_a,sent_b,_,_,targets,_,_ = data
        sent_a = sent_a.to(device)
        sent_b = sent_b.to(device)
        targets = targets.to(device)
        similarity_scores, sent_a_attentions, sent_b_attentions = model(sent_a, sent_b)

        pen_term_a = attention_penalty_loss(sent_a_attentions,
                                            model.self_attention_config["penalty"] , 
                                            device)
            
        pen_term_b = attention_penalty_loss(sent_b_attentions,
                                            model.self_attention_config["penalty"] , 
                                            device)


        loss = F.mse_loss(similarity_scores, targets) + pen_term_a + pen_term_b
        running_loss = running_loss + (loss.item() * len(sent_a))
        total_sentence = total_sentence + len(sent_a)
        targets = targets.to('cpu')
        similarity_scores = similarity_scores.to('cpu')
        predicted_scores.extend(similarity_scores.tolist())
        true_targets.extend(targets.tolist())
        if i%10 == 0:
            sentence_to_print = "Epoch : "+str(epoch+1)+", Validation Iteration : "+str(i+1)+" , Loss : "+ str(loss.item())
            print(sentence_to_print)
    validation_loss = running_loss / total_sentence
    sentence_to_print = "Epoch : "+str(epoch+1)+", Validation Loss : "+ str(validation_loss)
    print(sentence_to_print)
    s = torch.tensor(true_targets, dtype = torch.float32)
    ns = torch.tensor(predicted_scores, dtype = torch.float32, requires_grad = True)
    mse = F.mse_loss(ns, s)
    mes = mse.item()
    sp_cor, _ = stats.spearmanr(predicted_scores,true_targets)
    pr_cor, _ = stats.pearsonr(predicted_scores,true_targets)

    return  sp_cor, validation_loss

def attention_penalty_loss(annotation_weight_matrix, penalty_coef, device):
    """
    This function computes the loss from annotation/attention matrix
    to reduce redundancy in annotation matrix and for attention
    to focus on different parts of the sequence corresponding to the
    penalty term 'P' in the ICLR paper
    ----------------------------------
    'annotation_weight_matrix' refers to matrix 'A' in the ICLR paper
    annotation_weight_matrix shape: (batch_size, attention_out, seq_len)
    """
    batch_size, attention_out_size = annotation_weight_matrix.size(
        0
    ), annotation_weight_matrix.size(1)
    
    # TODO implement
    AAT = torch.bmm(annotation_weight_matrix, annotation_weight_matrix.transpose(1,2))
    I = torch.eye(attention_out_size).unsqueeze(0).repeat(batch_size, 1, 1)
    AAT = AAT.to(device)
    I = I.to(device)
    temp = AAT - I
    norm = frobenius_norm(temp)
    penalization_term = penalty_coef*(norm / batch_size)
    return penalization_term


def frobenius_norm(annotation_mul_difference):
    """
    Computes the frobenius norm of the annotation_mul_difference input as matrix
    """
    # TODO implement
    norm = torch.norm(annotation_mul_difference)
    return norm