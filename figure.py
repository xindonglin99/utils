import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import argparse

p = argparse.ArgumentParser()
p.add_argument('-p','--path', required=True, type=str, nargs='*', help='The file paths.')
args = p.parse_args()

def moving_avg(loss, window):
    loss_avg = []
    i = 0
    while i < len(loss) - window + 1:
        tmp = round(np.sum(loss[
        i:i+window]) / window, 2)
        loss_avg.append(tmp)
        i += 1
    return loss_avg

def main():
    file_paths = args.path
    print(file_paths)
    num_files = len(file_paths)
    data_arr = []
    for file_path in file_paths:
        data = pd.read_csv(file_path, usecols=['Step', 'Value']).to_numpy()
        print(data)
        data_arr.append(np.transpose(data, [1,0]))
    data_arr = np.array(data_arr)
    print(data_arr.shape)
    # plt.plot(data[0][0], data[0][1])
    # plt.plot(data[1][0], data[1][1])
    # plt.plot(data[2][0], data[2][1])
    # plt.plot(data[3][0], data[3][1])
    for data in data_arr:
        plt.plot(data[0], data[1])
    plt.legend(['BACON', 'SIREN', 'SAPE', 'LSLayer', 'FFN'])
    plt.xlim(0, 15000)
    plt.ylim(0, 0.03)
    plt.show()
    return 0

if __name__=='__main__':
    exit(main())