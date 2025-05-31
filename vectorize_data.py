import gensim
import manage_csv
import tokenize_data

def create_corpus(tokenized_data):
    corpus = []
    for tokens,category in tokenized_data:
        corpus.append(tokens)
    return corpus

if __name__ == "__main__":
    entry_list = manage_csv.create_data_entry_list("dev.csv", reduced=True, strip_category=True)
    tokenized_data = tokenize_data.tokenize(entry_list)
    corpus = create_corpus(tokenized_data)
    model = gensim.models.Word2Vec(corpus, min_count=1, vector_size=512, window=5)
    print(model.wv.most_similar('computer', topn=10))