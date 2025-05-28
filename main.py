"""
The purpose of this file is to have the main training/evaluation loop

"""


"""
Where model is a torch nn
Where set is the point pairs to evaluate on.
"""
def eval(model, pairs):
    num_correct = 0
    num_pos_correct = 0
    num_pred = 0
    num_gold = 0
    num_total = 0
    
    for x, y in pairs:
        pred = model.forward(x)
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
    