# SemanticTextSimilarity

This project is distributed among the three tasks :

# Task 1
In this task we combine architectures from two papers (Mueller & Thyagarajan,
2016) and (Lin et al., 2017) to solve the STS task. It forms the baseline for the
following tasks. Hence our network architecture look as follows:

![Task1](https://user-images.githubusercontent.com/94236355/177093288-106d5ffa-8da3-4efe-a4c8-8d474699b9ce.PNG)

# Task 2
In this task, we enhances our baseline architecture from task 1 by implementing the
Transformer Encoder (Vaswani et al., 2017) from scratch and adding to the architecture
from task 1. Hence our network architecture look as follows:

![image](https://user-images.githubusercontent.com/94236355/177093053-bd0a51bf-7562-42ff-88dc-7c5e5571e74d.png)

# Task 3
In this task, we use pretrained BERT ( bert-base-uncased) model which is different from models in
Task 1 and Task 2 to determine semantic textual similarity. BERT consists of 12 layers of tranformer
encoder. It is pretrained on a large corpus of English data (Wikipedia) in a self-supervised fashion.
We did not perform preprocessing on the data since the model itself will lowercase the input data.
Hence our network architecture look as follows:

![Task3_sig](https://user-images.githubusercontent.com/94236355/177093587-85fddff3-849d-4a17-8736-5c1ef3cb2c4a.PNG)


