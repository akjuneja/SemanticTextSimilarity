import pandas as pd
from preprocess import Preprocess
import logging
import torch
from dataset import STSDataset
from datasets import load_dataset
import torchtext
import re
import spacy
from torchtext.legacy.data import Field

import numpy as np
import random
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)


"""
For loading STS data loading and preprocessing
"""


class STSData:
    def __init__(
        self,
        dataset_name,
        columns_mapping,
        stopwords_path="stopwords-en.txt",
        model_name="lstm",
        max_sequence_len=15,
        normalization_const=5.0,
        normalize_labels=False,
    ):
        """
        Loads data into memory and create vocabulary from text field.
        """
        self.normalization_const = normalization_const
        self.normalize_labels = normalize_labels
        self.model_name = model_name
        self.max_sequence_len = max_sequence_len
        self.dataset_name = dataset_name
        ## load data file into memory
        self.load_data(dataset_name, columns_mapping, stopwords_path)
        self.columns_mapping = columns_mapping
        ## create vocabulary
        self.eng_tokenizer = spacy.load("en_core_web_sm")
        self.create_vocab()


    def load_data(self, dataset_name, columns_mapping, stopwords_path):
        """
        Reads data set file from disk to memory using pandas
        """
        logging.info("loading and preprocessing data...")
 
        ## TODO load datasets
        dataset_train, dataset_validation, dataset_test = load_dataset(self.dataset_name, split=['train', 'validation', 'test'])
        
        # removing unecessary columns
        columns_to_remove = [x for x in dataset_train.column_names if x not in columns_mapping.values()]

        #creating pd dataframe for train, validation, test
        dataset_train = dataset_train.remove_columns(columns_to_remove)
        data_train = pd.DataFrame(dataset_train)
        
        dataset_validation = dataset_validation.remove_columns(columns_to_remove)
        data_validation = pd.DataFrame(dataset_validation)

        dataset_test = dataset_test.remove_columns(columns_to_remove)
        data_test = pd.DataFrame(dataset_test)


        ## TODO perform text preprocessing
        preproc = Preprocess(stopwords_path)
        self.data_train = preproc.perform_preprocessing(data_train, columns_mapping)
        self.data_validation = preproc.perform_preprocessing(data_validation, columns_mapping)
        self.data_test = preproc.perform_preprocessing(data_test, columns_mapping)

        print("train-",len(self.data_train))
        print(self.data_train.loc[1:5,])

        logging.info("reading and preprocessing data completed...")

    def create_vocab(self):
        """
        Creates vocabulary over entire text data field.
        """
        logging.info("creating vocabulary...")

        # TODO create vocabulary
        
        data_train_concat = pd.DataFrame()
        data_train_concat["sentence_AB"] = self.data_train["sentence_A"] + self.data_train["sentence_B"]

        text_field = Field(tokenize='basic_english', lower=True)
        
        # applying  preprocess
        preprocessed_text = data_train_concat["sentence_AB"].apply(lambda x: text_field.preprocess(x))

        # load fastext simple embedding with 300d
        text_field.build_vocab(
            preprocessed_text, 
            vectors='fasttext.simple.300d')
        
        self.vocab = text_field.vocab

        logging.info("creating vocabulary completed...")

    def data2tensors(self, data):
        """
        Converts raw data sequences into vectorized sequences as tensors
        """
        pass
        # TODO implement
        sent_a = torch.tensor(data["sentence_A"].values.tolist())
        sent_b = torch.tensor(data["sentence_B"].values.tolist())
        scores = torch.tensor(data["relatedness_score"].values.tolist())
        
        return sent_a, sent_b, scores

    def get_data_loader(self, batch_size=8):
        pass
        # TODO implement
        data_loaders = dict()
        # train data
        data_train = self.data_train
        temp_train = data_train.copy()
        data_train["sentence_A"] = data_train["sentence_A"].apply(lambda x: self.vectorize_sequence(x))
        data_train["sentence_B"] = data_train["sentence_B"].apply(lambda x: self.vectorize_sequence(x))
        data_train['relatedness_score'] = (data_train['relatedness_score'] / self.normalization_const)
        data_train["sentence_A"] = data_train["sentence_A"].apply(lambda x: self.pad_sequences(x, self.max_sequence_len))
        data_train["sentence_B"] = data_train["sentence_B"].apply(lambda x: self.pad_sequences(x, self.max_sequence_len))
        sent_a, sent_b, scores = self.data2tensors(data_train)
        # create the train dataset and loader
        dataset_train = STSDataset(sent_a, sent_b, scores, sent_a, sent_b,
                                temp_train["sentence_A"].values.tolist(), temp_train["sentence_B"].values.tolist())
        
        data_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, drop_last=True)
        data_loaders["train"] = data_loader_train
        
        # validation data
        data_validation = self.data_validation
        temp_validation = data_validation.copy()
        data_validation["sentence_A"] = data_validation["sentence_A"].apply(lambda x: self.vectorize_sequence(x))
        data_validation["sentence_B"] = data_validation["sentence_B"].apply(lambda x: self.vectorize_sequence(x))
        data_validation['relatedness_score'] = (data_validation['relatedness_score'] / self.normalization_const)
        data_validation["sentence_A"] = data_validation["sentence_A"].apply(lambda x: self.pad_sequences(x, self.max_sequence_len))
        data_validation["sentence_B"] = data_validation["sentence_B"].apply(lambda x: self.pad_sequences(x, self.max_sequence_len))
        sent_a, sent_b, scores = self.data2tensors(data_validation)
        # create the train dataset and loader
        dataset_validation = STSDataset(sent_a, sent_b, scores, sent_a, sent_b,
                                temp_validation["sentence_A"].values.tolist(), temp_validation["sentence_B"].values.tolist())   
        data_loader_validation = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, drop_last=True)
        data_loaders["validation"] = data_loader_validation

        # test data
        data_test = self.data_test
        temp_test = data_test.copy()
        data_test["sentence_A"] = data_test["sentence_A"].apply(lambda x: self.vectorize_sequence(x))
        data_test["sentence_B"] = data_test["sentence_B"].apply(lambda x: self.vectorize_sequence(x))
        data_test['relatedness_score'] = (data_test['relatedness_score'] / self.normalization_const)
        data_test["sentence_A"] = data_test["sentence_A"].apply(lambda x: self.pad_sequences(x, self.max_sequence_len))
        data_test["sentence_B"] = data_test["sentence_B"].apply(lambda x: self.pad_sequences(x, self.max_sequence_len))
        sent_a, sent_b, scores = self.data2tensors(data_test)
        # create the test dataset and loader
        dataset_test = STSDataset(sent_a, sent_b, scores, sent_a, sent_b,
                                temp_test["sentence_A"].values.tolist(), temp_test["sentence_B"].values.tolist())
           
        data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, drop_last=True)
        data_loaders["test"] = data_loader_test
        
        return data_loaders


       
    def sort_batch(self, batch, targets, lengths):
        """
        Sorts the data, lengths and target tensors based on the lengths
        of the sequences from longest to shortest in batch
        """
        sents1_lengths, perm_idx = lengths.sort(0, descending=True)
        sequence_tensor = batch[perm_idx]
        target_tensor = targets[perm_idx]
        return sequence_tensor.transpose(0, 1), target_tensor, sents1_lengths

    def vectorize_sequence(self, sentence):
        """
        Replaces tokens with their indices in vocabulary
        """
        pass
        # TODO implement

        indices = []
        tokens = self.eng_tokenizer(sentence)

        for token in tokens:
            indices.append(self.vocab[token.text])
        
        return indices




    def pad_sequences(self, vectorized_sents, sents_lengths):
        """
        Pads zeros at the end of each sequence in data tensor till max
        length of sequence in that batch
        """
        # TODO implement
        padded_data = [0] * sents_lengths
        padded_data[0:len(vectorized_sents)] = vectorized_sents

        return padded_data
