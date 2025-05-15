import random
import sys
import manage_csv

'''
usage: split_data.py in out_train out_dev percent_train

Reads the input csv file <in>, splits data into train/dev sets
according to given percentage as float, and outputs these 
sets into the files <out_train> and <out_dev>.
'''

if __name__ == "__main__":
    if (len(sys.argv) != 5):
        print("usage: split_data.py in out_train out_dev percent_train")
        exit(1)
    input_filename = sys.argv[1]
    train_filename = sys.argv[2]
    dev_filename = sys.argv[3]
    train_portion = float(sys.argv[4])

    lst = manage_csv.create_data_entry_list(input_filename)
    random.shuffle(lst)

    dev_idx = int(len(lst) * train_portion)
    train_list = lst[:dev_idx]
    dev_list = lst[dev_idx:]

    manage_csv.create_csv_from_entries(train_list, train_filename)
    manage_csv.create_csv_from_entries(dev_list, dev_filename)
