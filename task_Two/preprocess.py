import re
import pandas as pd

"""
Performs basic text cleansing on the unstructured field 
"""


class Preprocess:
    def __init__(self, stpwds_file_path):
        """
        Initializes regex patterns and loads stopwords
        """
        # TODO implement
        stopwords = pd.read_csv(stpwds_file_path, header=None)
        self.stopwords_list = list(stopwords.loc[:,0])
        pass

    def perform_preprocessing(self, data, columns_mapping):

        for ind in data.index:
            sentence_A = data.loc[ind, columns_mapping["sent1"]]
            sentence_B = data.loc[ind, columns_mapping["sent2"]]

            ## TODO normalize text to lower case
            sentence_A = sentence_A.lower()
            sentence_B = sentence_B.lower()

            ## TODO remove punctuations
            sentence_A = re.sub(r'[^\w\s]',"",sentence_A)
            sentence_B = re.sub(r'[^\w\s]',"",sentence_B)

            ## TODO remove stopwords
            data.loc[ind, columns_mapping["sent1"]] = ' '.join([word for word in sentence_A.split() if word not in (self.stopwords_list)])
            data.loc[ind, columns_mapping["sent2"]] = ' '.join([word for word in sentence_B.split() if word not in (self.stopwords_list)])
        
            ## TODO add any other preprocessing method (if necessary)
            pass

        return data
