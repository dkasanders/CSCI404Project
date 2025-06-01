"""
The purpose of this file is to have the main training/evaluation loop

"""


import sys 
import numpy as np
import torch
from torch import nn
from neuralnetwork import DeepNeuralNet
from logisticregression import LogisticRegression
import manage_csv
import gensim
import tokenize_data


"""
Where model is a torch nn
Where set is the point pairs to evaluate on.
"""
def evaluate_model(model, pairs):
    model.eval()
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    
    for x, y in pairs:
        x = x.unsqueeze(0)
        pred = torch.argmax(model.forward(x)).item()
        if pred == y:
            num_correct += 1
        if pred == 1:
            num_pred += 1
            if y == 1:
                num_pos_correct += 1
        if y == 1:
            num_gold += 1
        num_total += 1

    precision = num_pos_correct / num_pred
    recall = num_pos_correct / num_gold
    accuracy = num_correct / num_total
    f1 = 2 * precision * recall / (precision + recall) 
    print(f"Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")




# TRAINING VARIABLES
EPOCHS = 100
LEARNING_RATE = 0.01
BATCH_SIZE = 1



#Prepare training and development data
#This currently is using the gensim word2vec model for vectorization of features
train_entries =  manage_csv.create_data_entry_list("train.csv", reduced=True, strip_category=True)
dev_entries = manage_csv.create_data_entry_list("dev.csv", reduced=True, strip_category=True)

def create_dataset(entries, model):
    result = []
    tokenized = tokenize_data.tokenize(entries)
    for tokens, category in tokenized:
        if len(tokens) == 0:
            continue
        vectors = [model.wv[tok] for tok in tokens if tok in model.wv]
        if not vectors:
            continue
        avg_vec = torch.tensor(np.mean(vectors, axis=0), dtype=torch.float32)
        label = torch.tensor(int(category), dtype=torch.long)
        result.append((avg_vec, label))
    return result

#Corpus and word2vec model
tokenized_train = tokenize_data.tokenize(train_entries)
corpus = [tokens for tokens, _ in tokenized_train]
w2v_model = gensim.models.Word2Vec(corpus, min_count=1, vector_size=64, window=15)

training = create_dataset(train_entries, w2v_model)
development = create_dataset(dev_entries, w2v_model)

#model definition
input_dim = 64
output_dim = len(set(e.category for e in train_entries))
model = LogisticRegression(input_dim, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE) 
criterion = torch.nn.CrossEntropyLoss(reduction='mean')




#Training loop
for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()
    for x, y in training:
        x = x.unsqueeze(0)
        y_pred = model(x)
        loss = criterion(y_pred, y.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss}")
    evaluate_model(model, development)




