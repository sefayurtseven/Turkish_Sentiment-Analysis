import numpy as np
import matplotlib.pyplot as plt


class PlotDraw(object):

    def __init__(self):
        print("Plot_Drawing Class")

    def DrawBarPlot(self, data_dict):
        data = {}
        for key in data_dict.keys():
            if data_dict[key] > 400:
                data[key] = data_dict[key]
        data_x = list(data.values())
        data_y = list(data.keys())
        x_pos = [i for i, _ in enumerate(data_y)]
        plt.barh(x_pos, data_x, color='black')
        plt.ylabel("Unique Words Over 400")
        plt.xlabel("Unique Words Frequency")
        plt.title("Corpus Unique Words Frequency Over 400 Graph ")

        plt.yticks(x_pos, data_y)

        plt.show()





