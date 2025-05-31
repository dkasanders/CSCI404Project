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

def tokenize(entry_list, category_map):
    '''
    convert data_entry list into list of tuples
    (tokens, category).

    tokens is a list of token strings and category is
    an integer.
    '''
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
    tokenized = tokenize(entry_list, cat_map)
    for i in range(10):
        print(f"ENTRY {i}")
        print(tokenized[i][0])