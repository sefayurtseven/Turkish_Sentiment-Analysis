import matplotlib
import numpy as np
import matplotlib.pyplot as plt


class PlotDraw(object):

    def __init__(self):
        print("Plot_Drawing Class")

    def DrawBarPlot(self, data_dict):
        font = {'family': 'normal',
                'weight': 'bold',
                'size': 16}

        matplotlib.rc('font', **font)
        data = {}
        for key in list(data_dict.keys())[:10]:
            data[key] = data_dict[key]
        data_x = list(data.values())
        data_y = list(data.keys())
        x_pos = [i for i, _ in enumerate(data_y)]
        plt.barh(x_pos, data_x, color='black')
        plt.ylabel("Negative Word", fontsize=18)
        plt.xlabel("Negative Word Frequency", fontsize=18)
        plt.title("Negative Most Frequent 10 Word", fontsize=20)

        plt.yticks(x_pos, data_y)

        plt.show()





