import csv

redir = {
    'cond-mat': 'physics',
    'astro-ph': 'physics',
    'nlin': 'physics',
    'quant-ph': 'physics',
    'hep-th': 'physics',
    'nucl-ex': 'physics',
    'gr-qc': 'physics',
    'math-ph': 'physics',
    'nucl-th': 'physics',
    'comp-gas': 'physics',
    'hep-lat': 'physics',
    'hep-ex': 'physics',
    'hep-ph': 'physics'
}

class data_entry:
    '''
    ATTRIBUTES
    
    id: str
    title: str
    summary: str
    author: str
    category: str
    '''

    def __init__(self, id: str, title: str, summary: str, author: str, category: str):
        self.id = id
        self.title = title
        self.summary = summary
        self.author = author
        self.category = category

def create_data_entry_list(filename, reduced=False, strip_category=False):
    '''
    create and return list of data_entry objects based on csv file with name <filename>.

    by default, the file is expected to have the following columns:
    [row_num], id, Title, Summary, Author, Link, Publish Date, Update Date, Primary Category, Category

    if reduced is set to True, the file is expected to have the following columns:
    [row_num], id, title, summary, category
    '''

    file = open(filename, encoding="utf-8", newline='')
    reader = csv.reader(file)
    if (reduced):
        atts = {    # map attributes used for data_entry to their indices in each csv row
            "id": 1,
            "title": 2,
            "summary": 3,
            "author": 4,
            "category": 5
        }
    else:
        atts = {   
            "id": 1,
            "title": 2,
            "summary": 3,
            "author": 4,
            "category": 8
        }

    result = []
    for row in reader:
        curr_entry = data_entry(row[atts["id"]], row[atts["title"]], row[atts["summary"]], row[atts["author"]], row[atts["category"]])
        if strip_category:
            curr_entry.category = curr_entry.category.split(".")[0]
            if curr_entry.category in redir:
                curr_entry.category = redir[curr_entry.category]
        result.append(curr_entry)
    file.close()
    return result[1:]

def create_csv_from_entries(entry_list: list, output_file_name: str):
    '''
    create new csv file based on given list of entry objects <entry_list>.
    output file will have name <output_file_name>.

    the rows in the resulting file will have the following attributes:
    id, title, summary, author, category
    '''

    output_file = open(output_file_name, "w", newline='', encoding="utf-8")
    writer = csv.writer(output_file)

    writer.writerow(['','id','title','summary','author','category'])
    
    row_number = 1
    for e in entry_list:
        writer.writerow([row_number, e.id, e.title, e.summary, e.author, e.category])
        row_number += 1
    output_file.close()



if __name__ == "__main__":  # for testing/debugging
    lst = create_data_entry_list('dev.csv', reduced=True, strip_category=True)
    print(lst[1].category)
