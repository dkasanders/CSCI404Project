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
from bertopic import BERTopic

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



#Word2vec embedding creation
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
#tokenized_train = tokenize_data.tokenize(train_entries)
#corpus = [tokens for tokens, _ in tokenized_train]
#w2v_model = gensim.models.Word2Vec(corpus, min_count=1, vector_size=64, window=15)

#BERTopic embeddings
def create_dataset_BERT(entries, topic_model):
    result = []
    print(vars(entries[0]))
    docs = [e.title + "!!DIV!!" + e.summary for e in entries]
    _, probs = topic_model.transform(docs)
    for prob, entry in zip(probs, entries):
        vec = torch.tensor(prob, dtype=torch.float32)
        label = torch.tensor(int(entry.category), dtype=torch.long)
        result.append((vec, label))
    return result


train_text_label_pairs, category_map = tokenize_data.raw_text_and_label(train_entries)
dev_test_label_pairs, _ = tokenize_data.raw_text_and_label(dev_entries)

docs_train = [text for text, _ in train_text_label_pairs]
labels_train = [label for _, label in train_text_label_pairs]

docs_dev = [text for text, _ in dev_test_label_pairs]
labels_dev = [label for _, label in dev_test_label_pairs]


bertopic_model = BERTopic.load("trained_model")
#topics_train, probs_train = bertopic_model.fit_transform(docs_train, y=labels_train)
#bertopic_model.save("trained_model")

topics_train, probs_train = bertopic_model.transform(docs_train)
topics_dev, probs_dev = bertopic_model.transform(docs_dev)

training = create_dataset_BERT(train_entries, bertopic_model)
development = create_dataset_BERT(dev_entries, bertopic_model)

#model definition
input_dim = len(training[0][0])
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




