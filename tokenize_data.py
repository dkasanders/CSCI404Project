import nltk 
# nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

text = "This is an example of a sentence to be tokenized"

print(word_tokenize(text))
