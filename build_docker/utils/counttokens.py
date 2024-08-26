import os
import sys
import pickle

def read_pickle(file_fp):
    with open(file_fp, "rb") as handler:
        return pickle.load(handler)


def main():

    #main dist between 10 and 33
    sizes_dict = read_pickle("sizes_dict.pkl")
    x_axis = list()
    y_axis = list()
    for key in sorted(sizes_dict.keys()):
        x_axis.append(key)
        y_axis.append(sizes_dict[key])
    import pdb
    pdb.set_trace()

if __name__ == "__main__":
    main()