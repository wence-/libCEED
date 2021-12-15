#!/usr/bin/env python3

import pandas as pd
import argparse
from pylab import *
from matplotlib import use


def plot():
    # Define argparse for the input variables
    parser = argparse.ArgumentParser(description='Get input arguments')
    parser.add_argument('-f',
                        dest='conv_result_file',
                        type=str,
                        required=True,
                        help='Path to the CSV file')
    args = parser.parse_args()
    conv_result_file = args.conv_result_file

    # Load the data
    runs = pd.read_csv(conv_result_file)
    colors = ['orange', 'red', 'navy', 'green', 'magenta',
              'gray', 'blue', 'purple', 'pink', 'black']
    res = 'mesh_res'
    fig, ax = plt.subplots()

    i = 0
    for group in runs.groupby('degree'):
        data = group[1]
        data = data.sort_values('rel_error')
        p = data['degree'].values[0]
        h = 1/data[res]
        E = data['rel_error']
        H =  amin(E) * (h/amin(h))**p
        ax.loglog(h, E, 'o', color=colors[i])
        ax.loglog(h, H, '--', color=colors[i], label='O(h$^' + str(p) + '$)')
        i = i + 1

    ax.legend(loc='best')
    ax.set_xlabel('h')
    ax.set_ylabel('Relative Error')
    ax.set_title('Convergence by h Refinement')
    plt.savefig('conv_plt_h.png')


if __name__ == "__main__":
    plot()
