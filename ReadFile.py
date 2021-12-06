import codecs
import csv
import pandas as pd


class ReadFile(object):
    file_path = ""

    def __init__(self, file_path):
        print("ReadFile Class")
        self.file_path = file_path

    def read_csv_file(self, file_path):
        with open(file_path, 'r') as file:
            reader = csv.reader(codecs.open(file_path, 'rU', 'utf-16'))
            for row in reader:
                print(row)

    def read_csv_via_pandas(self):
        data_frame = pd.read_csv(codecs.open(self.file_path, 'rU', 'utf-16'))
        data_frame = data_frame.dropna()
        return data_frame

    def read_txt_file(self):
        with open(self.file_path, "r", encoding="utf8") as f:
            return f.read().split("\n")