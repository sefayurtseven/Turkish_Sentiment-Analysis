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
import pickle
import json


if __name__ == '__main__':
    dir_name = os.path.dirname(__file__)
    dataset_path = dir_name + "\\dataset\\magaza_yorumlari.csv"
    stop_words_path = dir_name + "\\Turkish_Stop_Words.txt"

    # read_csv_file = ReadFile.ReadFile(dataset_path)
    # stop_words_file = ReadFile.ReadFile(stop_words_path)
    # text_process = Text_Pre_Processing.TextPreProcessing()
    #
    # stop_words_list = stop_words_file.read_txt_file()
    # stop_words_list = text_process.unique_vowels_change(stop_words_list)
    #
    # data_frame = read_csv_file.read_csv_via_pandas()
    # data_frame['Görüş'] = text_process.lower_case_apply(data_frame['Görüş'])
    # data_frame['unique_letter_changed_sentence'] = text_process.unique_vowels_change(data_frame['Görüş'])
    # data_frame['tokenized_sents'] = text_process.tokenize_sentences(data_frame['Görüş'])
    # data_frame['stop_words_removed'] = text_process.remove_stop_words(data_frame['tokenized_sents'], stop_words_list)
    # data_frame['stemming_applied'] = text_process.stemming(data_frame['stop_words_removed'])
    #
    # # print(data_frame)
    # with open('filename_dataframe.pkl', 'wb') as outp:
    #     pickle.dump(data_frame, outp, pickle.HIGHEST_PROTOCOL)

    with open('filename_dataframe.pkl', 'rb') as inp:
        data_frame_pkl = pickle.load(inp)
        print(data_frame_pkl)  # -> banana
    lamma_dict = {}
    for lemma_list in data_frame_pkl['stemming_applied']:
        print(lemma_list)
        for l in lemma_list:
            if len(l) >= 1:
                if lamma_dict.keys().__contains__(l[0][0]):
                    for lemma in l[0][1]:
                        if lamma_dict[l[0][0]].keys().__contains__(lemma):
                            lamma_dict[l[0][0]][lemma] +=1
                        else:
                            lamma_dict[l[0][0]][lemma] = 1
                else:
                    lamma_dict[l[0][0]] = {}
                    for lemma in l[0][1]:
                        lamma_dict[l[0][0]][lemma] = 1
    with open("lemmatization.json", "w") as outfile:
        json.dump(lamma_dict, outfile)

print("sefa")
   # print(stop_word_list)


