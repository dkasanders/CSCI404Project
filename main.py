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
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

"""
This takes the entries and constructs a map that maps the index to its categorial value
"""
def build_category_map(entries):
    category_values = sorted(set(e.category for e in entries))
    category_to_index = {cat: idx for idx, cat in enumerate(category_values)}
    return {idx: cat for cat, idx in category_to_index.items()}


"""
Where model is a torch nn
Where set is the point pairs to evaluate on.
Index_to_category is a mapping from index in model guesses to name of the category
"""
def evaluate_model(model, pairs, index_to_category, print_metrics):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    
    #For tracking incorrect positions, where it maps {true_label: {predicted label: count}}
    confusion_tracker = defaultdict(lambda: defaultdict(int))
    #For tracking which category isn't guessed correctly the most, {true_label: number of times not guessed correctly}
    incorrect_count = defaultdict(int)

    for x, y in pairs:
        x = x.unsqueeze(0)
        y_tensor = torch.tensor([y])

        model_output = model(x)
        pred = torch.argmax(model_output).item()

        total_loss += loss_fn(model_output, y_tensor).item()
        if pred == y:
            num_correct += 1
        else:
            confusion_tracker[y.item()][pred] += 1
            incorrect_count[y.item()] += 1
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
    print(f"Loss: {total_loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    if(print_metrics):
        print("Incorrect count per true category:")
        for true_label, count in sorted(incorrect_count.items()):
            label_str = index_to_category[true_label]
            print(f"{label_str}: mislabeled {count} time(s)")

        print("Incorrect Predicitions Breakdown:")
        for true_label in sorted(confusion_tracker):
            true_label_str = index_to_category[true_label]
            print(f"True Category {true_label_str}:")
            for pred_label, count in sorted(confusion_tracker[true_label].items()):
                pred_label_str = index_to_category[pred_label]
                print(f"    Predicted as {pred_label_str}: {count} time(s)")




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

#Simple baseline model that just guesses the highest probability topic for each embedding
#Bertopic constructs embeddings where each index is the probability of that class.
def baseline_bertopic_guess(embedding):
    return int(torch.argmax(embedding))


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



bertopic_model = BERTopic(calculate_probabilities=True)
topics_train, probs_train = bertopic_model.fit_transform(docs_train, y=labels_train)
bertopic_model.save("trained_model")

#topics_train, probs_train = bertopic_model.transform(docs_train, calculate_probabilities=True)
topics_dev, probs_dev = bertopic_model.transform(docs_dev)

training = create_dataset_BERT(train_entries, bertopic_model)
development = create_dataset_BERT(dev_entries, bertopic_model)


training = []
for prob,label in zip(probs_train, labels_train):
    x = torch.tensor(prob, dtype=torch.float32)
    y = torch.tensor(int(label), dtype=torch.long)
    training.append((x, y))

development = []
for prob, label in zip(probs_dev, labels_dev):
    x = torch.tensor(prob, dtype=torch.float32)
    
    y = torch.tensor(int(label), dtype=torch.long)
    
    development.append((x, y))

torch.save(training, "training_set_BERT.pt")
torch.save(development, "development_set_BERT.pt")

training = torch.load("training_set_BERT.pt")
development = torch.load("development_set_BERT.pt")

index_to_category = build_category_map(train_entries)


#model definition

#These are for the w2v encoding scheme
#input_dim = len(training[0][0])
#output_dim = len(set(e.category for e in train_entries))

input_dim = 4
output_dim = 3

model = LogisticRegression(input_dim, output_dim)
optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE) 
criterion = torch.nn.CrossEntropyLoss(reduction='mean')


loss_fn = nn.CrossEntropyLoss()

total_loss = 0.0
    
num_correct = 0
num_pos_correct = 0
num_pred = 0
num_gold = 0
num_total = 0
    
#For tracking incorrect positions, where it maps {true_label: {predicted label: count}}
confusion_tracker = defaultdict(lambda: defaultdict(int))
#For tracking which category isn't guessed correctly the most, {true_label: number of times not guessed correctly}
incorrect_count = defaultdict(int)

y_true = []
y_pred = []

for x, y in development:
    print(x)
    pred = torch.argmax(x).item()
   
    true = y
    y_pred.append(pred)
    y_true.append(true)

    if pred == true:
        print(f"this happened! pred is {pred}")
        num_correct += 1
    num_total += 1

#print(y_true)
#print(y_pred)

precision = precision_score(y_true, y_pred, average='micro')
recall = recall_score(y_true, y_pred, average='micro')
accuracy = num_correct / num_total
f1 = f1_score(y_true, y_pred, average='micro')

print(f"Loss: {total_loss:.4f}, Acc: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

print("Incorrect count per true category:")
for true_label, count in sorted(incorrect_count.items()):
    label_str = index_to_category[true_label]
    print(f"{label_str}: mislabeled {count} time(s)")

print("Incorrect Predicitions Breakdown:")
for true_label in sorted(confusion_tracker):
    true_label_str = index_to_category[true_label]
    print(f"True Category {true_label_str}:")
    for pred_label, count in sorted(confusion_tracker[true_label].items()):
        pred_label_str = index_to_category[pred_label]
        print(f"    Predicted as {pred_label_str}: {count} time(s)")





#Training loop
for epoch in range(EPOCHS):
    total_loss = 0.0
    model.train()
    for x, y in training:
        x = x.unsqueeze(0)
        y_pred = model(x)
        #y_pred = baseline_bertopic_guess(x)
        loss = criterion(y_pred, y.unsqueeze(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss}")
    evaluate_model(model, development, index_to_category, False)




