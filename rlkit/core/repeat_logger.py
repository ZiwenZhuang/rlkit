#! python3
''' This is an individual file that depends on only the following library.
'''
import numpy as np
import os
import os.path as osp
import sys
import datetime
import dateutil
import csv
import json
import pickle
import matplotlib.pyplot as plt

class RepeatLogger:
    ''' This object is designed to logging repeated experiment into csv file.
        It lets you compute the mean and variance later while reloading.
    '''
    def __init__(self, filename, allow_exist=True):
        ''' The filename has to be absolute path (for the safety).
            If you allow_exist, the object will append on the existing file if it exist,
                or it will create a new file with postfix on the filename
        '''
        self._filename = filename
        # create file
        try:
            self._fd = open(filename, mode='x', newline='')
        except OSError as e:
            if allow_exist:
                # add file creation time at the end of the filename (before .csv)
                _name, _postfix = filename.rsplit('.', 1)
                now = datetime.datetime.now(dateutil.tz.tzlocal())
                timestamp = now.strftime('%Y-%m-%d-%H:%M:%S')
                filename = '.'.join([_name, timestamp, _postfix])
                self._fd = open(filename, mode='x', newline='')
            else:
                raise e

    def __del__(self):
        if hasattr(self, '_fd'):
            self._fd.close()

    def record(self, data):
        ''' Record one row of data, for one experiment.
            It is better to be iterable, and can be written directly to csv file row.
        '''
        writer = csv.writer(self._fd, delimiter=' ', quotechar='|')
        writer.writerow(data)

class RepeatPlotter:
    ''' The co-reponding object to read file and plot them as image (or extract the data)
    '''
    def __init__(self, filename):
        ''' It is safer id you provide absolute path.
        '''
        self._filename = filename
        self._fd = open(filename, mode='r')
        
    def get_data(self):
        ''' Return several statics of the sequence of data. A dictionary with defined keywords
        '''
        reader = csv.reader(self._fd, delimiter=' ', quotechar='|')
        
        self.raw_data = [] # it will be a np array when the function returns
        for row in reader:
            self.raw_data.append([float(num_string) for num_string in row])
        self.raw_data = np.array(self.raw_data) # assuming each row has the same length

        mean = np.mean(self.raw_data, axis=0)
        std = np.std(self.raw_data, axis=0)
        maxi = np.amax(self.raw_data, axis=0)
        mini = np.amin(self.raw_data, axis=0)

        return {
            'mean': mean,
            'std': std,
            'max': maxi,
            'min': mini,
        }

    def plot(self, options, smooth=True):
        ''' Plot the file data based on give options.
            options should be a list of string selected from the following:
                "mean", "std", "max", "min",
            If allowing smooth, the plot will take average on each 10 consecutive number.
        '''
        sorted_data = self.get_data()
        for choice in options:
            assert choice in sorted_data.keys()
            plt.plot(sorted_data[choice])

        plt.show()

def main(args):
    plotter = RepeatPlotter(args.file)

    plotter.plot(args.option)

if __name__ == '__main__':
    import argparse
    # run the plotter directly
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str,
                        help='the file to be extracted and plot')
    parser.add_argument('--option', type=str, action='append',
                        help='the figures you want to plot, choice from following: mean, std, max, min')
    args = parser.parse_args()

    main(args)
