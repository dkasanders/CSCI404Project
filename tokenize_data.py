import manage_csv
import nltk 
from nltk.tokenize import word_tokenize

def create_category_map(entry_list):
    '''
    using the given data_entry list, creates and returns
    a dictionary which maps all C distinct categories to
    a distinct integer id in the range [0,C-1]
    '''
    categories = set([entry.category for entry in entry_list])
    curr_label = 0
    result = dict()
    for cat in categories:
        result[cat] = curr_label
        curr_label += 1
    return result

'''
Converts entry_list into a list of (text, category_id) pairs
'''
def raw_text_and_label(entry_list):
    category_map = create_category_map(entry_list)
    result = []
    for entry in entry_list:
        full_text = entry.title + "!!DIV!!" + entry.summary
        label = category_map[entry.category]
        result.append((full_text, label))
    return result, category_map


def tokenize(entry_list):
    '''
    convert data_entry list into list of tuples
    (tokens, category).

    tokens is a list of token strings and category is
    an integer.
    '''
    category_map = create_category_map(entry_list)
    result = []
    for entry in entry_list:
        tokenized_title = word_tokenize(entry.title)
        tokenized_summary = word_tokenize(entry.summary)
        token_list = []
        token_list += tokenized_title
        token_list.append("!!DIV!!")
        token_list += tokenized_summary
        result.append((token_list, category_map[entry.category]))
    return result

if __name__ == "__main__":
    entry_list = manage_csv.create_data_entry_list("dev.csv", reduced=True, strip_category=True)
    cat_map = create_category_map(entry_list)
    tokenized = tokenize(entry_list)
    for i in range(10):
        print(f"ENTRY {i}")
        print(tokenized[i][0])
        
    print(word_tokenize("The cat jumps!"))