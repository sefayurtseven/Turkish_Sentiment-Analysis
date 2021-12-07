# https://towardsdatascience.com/text-preprocessing-in-natural-language-processing-using-python-6113ff5decd8
# https://www.analyticsvidhya.com/blog/2021/08/text-preprocessing-techniques-for-performing-sentiment-analysis/
# https://anilozbek.blogspot.com/2019/01/stemming-ve-lemmatization.html
# https://www.analyticsvidhya.com/blog/2021/08/text-preprocessing-techniques-for-performing-sentiment-analysis/
# https://towardsdatascience.com/updated-text-preprocessing-techniques-for-sentiment-analysis-549af7fe412a
# https://github.com/ahmetax/trstop/tree/master/dosyalar
# https://towardsdatascience.com/cleaning-preprocessing-text-data-for-sentiment-analysis-382a41f150d6
#
# https://www.revuze.it/blog/sentiment-analysis-a-step-by-step-guide-2021/
# https://realpython.com/sentiment-analysis-python/
# https://medium.com/@paritosh_30025/natural-language-processing-text-data-vectorization-af2520529cf7
# https://www.analyticsvidhya.com/blog/2021/06/part-5-step-by-step-guide-to-master-nlp-text-vectorization-approaches/
# https://www.analyticsvidhya.com/blog/2021/06/part-5-step-by-step-guide-to-master-nlp-text-vectorization-approaches/

import os
import ReadFile
import Text_Pre_Processing
import PlotDrawing
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    dataset_path = dir_name + "\\dataset\\magaza_yorumlari.csv"
    stop_words_path = dir_name + "\\Turkish_Stop_Words.txt"

    read_csv_file = ReadFile.ReadFile(dataset_path)
    stop_words_file = ReadFile.ReadFile(stop_words_path)
    text_process = Text_Pre_Processing.TextPreProcessing()
    plot_drawing = PlotDrawing.PlotDraw()

    stop_words_list = stop_words_file.read_txt_file()

    # data_frame = read_csv_file.read_csv_via_pandas()
    # train_dataset = data_frame[:5000]
    # train_dataset['Görüş'] = text_process.lower_case_apply(train_dataset['Görüş'])
    # train_dataset['Punctuation_Romoved'] = text_process.remove_punctuations(train_dataset['Görüş'])
    # train_dataset['tokenized_sents'] = text_process.tokenize_sentences(train_dataset['Punctuation_Romoved'])
    # train_dataset['stop_words_removed'] = text_process.remove_stop_words(train_dataset['tokenized_sents'], stop_words_list)
    # train_dataset['stemming_applied'] = text_process.sent_lemmatize(train_dataset['stop_words_removed'])
    # word_freq_dict = text_process.get_word_freq_dict(train_dataset['stemming_applied'])
    # for sw in stop_words_list:
    #     if word_freq_dict.keys().__contains__(sw):
    #         word_freq_dict.pop(sw)

    # plot_drawing.DrawBarPlot(word_freq_dict)

    # print(data_frame)
    # with open('filename_train_dataset.pkl', 'wb') as outp:
    #     pickle.dump(train_dataset, outp, pickle.HIGHEST_PROTOCOL)

    with open('filename_train_dataset.pkl', 'rb') as inp:
        data_frame_pkl = pickle.load(inp)
    # word_freq_dict = text_process.get_word_freq_dict(data_frame_pkl['stemming_applied'])
    count = 0
    # for key in word_freq_dict.keys().__reversed__():
    #     if word_freq_dict[key] == 1:
    #         count += 1
    df_status_negative = data_frame_pkl[data_frame_pkl.Durum == "Olumsuz"]
    df_status_positive = data_frame_pkl[data_frame_pkl.Durum == "Olumlu"]
    positive_bigram_list = []
    negative_bigram_list = []
    for sent in df_status_positive['stemming_applied']:
        bigram_list = []
        if len(sent) > 1:
            index = 0;
            while index +1 < len(sent):
                bigram_list.append([sent[index], sent[index + 1]])
                index += 1
        positive_bigram_list.append(bigram_list)
    for sent in df_status_negative['stemming_applied']:
        bigram_list = []
        if len(sent) > 1:
            index = 0;
            while index +1 < len(sent):
                bigram_list.append([sent[index], sent[index + 1]])
                index += 1
        negative_bigram_list.append(bigram_list)
    df_status_positive['bigram_lists'] = positive_bigram_list
    df_status_negative['bigram_lists'] = negative_bigram_list
    bigram_freq_dict_p = text_process.get_word_freq_dict(df_status_positive['bigram_lists'])
    bigram_freq_dict_n = text_process.get_word_freq_dict(df_status_negative['bigram_lists'])
    print()
    plot_drawing.DrawBarPlot(bigram_freq_dict_n)
    # word_freq_dict_positive  = text_process.get_word_freq_dict(df_status_positive['stemming_applied'])
    # word_freq_dict_negative  = text_process.get_word_freq_dict(df_status_negative['stemming_applied'])
    # plot_drawing.DrawBarPlot(word_freq_dict_negative)
    # d = {}
    # d['Positive'] = word_freq_dict_positive['güzel']
    # d['Negative'] = word_freq_dict_negative['güzel']
    # # creating the dataset
    # data = {'C': 20, 'C++': 15, 'Java': 30,
    #         'Python': 35}
    # courses = list(d.keys())
    # values = list(d.values())
    #
    # fig = plt.figure(figsize=(4, 3))
    #
    # # creating the bar plot
    # plt.bar(courses, values, color='black',
    #         width=0.75)
    #
    # plt.xlabel("Sentiment")
    # plt.ylabel("güzel")
    # plt.title("")
    # plt.show()
    print()

    # lamma_dict = {}
    # for lemma_list in data_frame_pkl['stemming_applied']:
    #     print(lemma_list)
    #     for l in lemma_list:
    #         if len(l) >= 1:
    #             if lamma_dict.keys().__contains__(l[0][0]):
    #                 for lemma in l[0][1]:
    #                     if lamma_dict[l[0][0]].keys().__contains__(lemma):
    #                         lamma_dict[l[0][0]][lemma] +=1
    #                     else:
    #                         lamma_dict[l[0][0]][lemma] = 1
    #             else:
    #                 lamma_dict[l[0][0]] = {}
    #                 for lemma in l[0][1]:
    #                     lamma_dict[l[0][0]][lemma] = 1
    # with open("lemmatization.json", "w") as outfile:
    #     json.dump(lamma_dict, outfile)

   # print(stop_word_list)


