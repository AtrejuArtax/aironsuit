import os
import numpy as np
import pandas as pd


def solution(files: list):
    # files - any of available files, i.e:
    # files = ["./data/framp.csv", "./data/gnyned.csv", "./data/gwoomed.csv",
    #            "./data/hoilled.csv", "./data/plent.csv", "./data/throwsh.csv",
    #            "./data/twerche.csv", "./data/veeme.csv"]

    # write your solution here
    data = []
    for file_name in files:
        data_ = pd.read_csv(file_name)  # Load csv file
        data_['date'] = pd.to_datetime(data_['date'])  # Date column to datetime type
        data_['year'] = data_['date'].dt.year  # Extract the year
        # New column with the highest closing per subgroup of year
        data_['highest_close'] = data_['year'].map(data_[['year', 'date', 'close']].groupby(['year'])['close'].max())
        data += [[
            data_[['year', 'date', 'vol']].groupby('year').max().reset_index()[['date', 'vol']],
            data_[['date', 'close']][data_['close'] == data_['highest_close']].reset_index()[['date', 'close']]
        ]]
        del data_
    return data


if __name__ == '__main__':

    path = '/home/claudi/Downloads/csv_files'
    file_names = [os.path.join(path, file_name) for file_name in os.listdir(path)]
    output = solution(file_names)
