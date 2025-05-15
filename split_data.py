import random
import sys
import manage_csv

'''
usage: split_data.py in out_train out_dev out_test percent_train percent_dev

Reads the input csv file <in>, splits data into train/dev/test sets
according to given percentages given as floats, and outputs these 
sets into the files <out_train>, <out_dev>, and <out_test>.

note: the test set percentage will be inferred as 1 - percent_train - percent_dev
'''

if __name__ == "__main__":
    if (len(sys.argv) != 7):
        print("usage: split_data.py in out_train out_dev out_test percent_train percent_dev")
        exit(1)
    input_filename = sys.argv[1]
    train_filename = sys.argv[2]
    dev_filename = sys.argv[3]
    test_filename = sys.argv[4]
    train_portion = float(sys.argv[5])
    dev_portion = float(sys.argv[6])

    lst = manage_csv.create_data_entry_list(input_filename)
    random.shuffle(lst)

    dev_idx = int(len(lst) * train_portion)
    test_idx = dev_idx + int(len(lst) * dev_portion)
    train_list = lst[:dev_idx]
    dev_list = lst[dev_idx:test_idx]
    test_list = lst[test_idx:]

    manage_csv.create_csv_from_entries(train_list, train_filename)
    manage_csv.create_csv_from_entries(dev_list, dev_filename)
    manage_csv.create_csv_from_entries(test_list, test_filename)
