import csv

class data_entry:
    '''
    ATTRIBUTES
    
    id: int
    title: str
    summary: str
    author: str
    category: str
    '''

    def __init__(self, id: int, title: str, summary: str, author: str):
        self.id = id
        self.title = title
        self.summary = summary
        self.author = author

def create_data_entry_list(filename):
    '''
    create and return list of data_entry objects based on csv file with name <filename>.

    the file is expected to have the following columns:
    id, Title, Summary, Author, Link, Publish Date, Primary Category, Category
    '''
    file = open(filename)
    reader = csv.reader(file)


if __name__ == "__main__":
    ...