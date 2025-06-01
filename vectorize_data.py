import gensim
import manage_csv
import tokenize_data

def create_corpus(tokenized_data):
    corpus = []
    for tokens,category in tokenized_data:
        corpus.append(tokens)
    return corpus

def vectorize(entry_list):
    tokenized_data = tokenize_data.tokenize(entry_list)
    corpus = create_corpus(tokenized_data)
    model = gensim.models.Word2Vec(corpus, min_count=1, vector_size=64, window=15)
    result = []
    for tokens,category in tokenized_data:
        vectorized_tokens = [model.wv[tok] for tok in tokens]
        result.append((vectorized_tokens, category))
    return result

if __name__ == "__main__":
    entry_list = manage_csv.create_data_entry_list("train.csv", reduced=True, strip_category=True)
    tokenized_data = tokenize_data.tokenize(entry_list)
    corpus = create_corpus(tokenized_data)
    model = gensim.models.Word2Vec(corpus, min_count=1, vector_size=64, window=15)
    print(model.wv.most_similar('political', topn=20))