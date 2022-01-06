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
# https://pythonprogramming.net/naive-bayes-classifier-nltk-tutorial/
# https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
# https://www.andreaperlato.com/mlpost/nlp-step-by-step/
import os
import token
import tokenize

import matplotlib
import nltk
from nltk.translate import metrics
from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler

import ReadFile
import Text_Pre_Processing
import PlotDrawing
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

def calculate_BOW(word_set, l_doc):
    tf_diz = dict.fromkeys(word_set, 0)
    for word in l_doc:
        tf_diz[word] = l_doc.count(word)
    return tf_diz
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

def computeIDF(documents):
    import math
    N = len(documents)
    d = []
    for i in list(documents[0].items()):
        d.append(i[0])
    idfDict = dict.fromkeys(d, 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1

    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

if __name__ == '__main__':
    # Initialize
    # dir_name = os.path.dirname(__file__)
    # dataset_path = dir_name + "\\dataset\\magaza_yorumlari.csv"
    # stop_words_path = dir_name + "\\Turkish_Stop_Words.txt"
    #
    # read_csv_file = ReadFile.ReadFile(dataset_path)
    # df = read_csv_file.read_csv_via_pandas()
    # stop_words_file = ReadFile.ReadFile(stop_words_path)
    # stop_words_list = stop_words_file.read_txt_file()
    #
    # # Text Processing
    # text_process = Text_Pre_Processing.TextPreProcessing()
    # df = df.rename(columns={'Görüş': 'Review', 'Durum': 'Sentiment'})
    # df['Review_Lower_Case'] = text_process.lower_case_apply(df['Review'])
    # df['Punctuation_Romoved'] = text_process.remove_punctuations(df['Review_Lower_Case'])
    # df['Tokenized_Sents'] = text_process.tokenize_sentences(df['Punctuation_Romoved'])
    # df['Stop_Words_Removed'] = text_process.remove_stop_words(df['Tokenized_Sents'], stop_words_list)
    # df['stemming_applied'] = text_process.sent_lemmatize(df['Stop_Words_Removed'])
    #
    # with open('dataset_new.pkl', 'wb') as outp:
    #     pickle.dump(df, outp, pickle.HIGHEST_PROTOCOL)

    with open('dataset_new.pkl', 'rb') as inp:
        df = pickle.load(inp)

    df_status_negative = pd.DataFrame(df[df.Sentiment == "Olumsuz"]).head(2)
    df_status_positive = pd.DataFrame(df[df.Sentiment == "Olumlu"]).head(2)
    df_2000 = pd.concat([df_status_positive, df_status_negative])



    wordset12 = np.union1d(df_2000['stemming_applied'][0], df_2000['stemming_applied'][1])
    for sent in df_2000['stemming_applied'][2:]:
        wordset12 = np.union1d(wordset12, sent)

    # with open('wordset12.pkl', 'wb') as outp:
    #     pickle.dump(wordset12, outp, pickle.HIGHEST_PROTOCOL)
    #
    # # with open('wordset12.pkl', 'rb') as inp:
    # #     wordset12 = pickle.load(inp)
    #
    # BOW_list = []
    # i = 0
    # for sent in df_2000['stemming_applied']:
    #     print(i)
    #     i = i + 1
    #     BOW_list.append(calculate_BOW(wordset12, sent))
    # df_bow = pd.DataFrame(BOW_list)
    # df_bow["Sentiment"] = df_2000["Sentiment"]
    # df_bow = df_bow.dropna()
    #
    # bagOfWords = df_bow[df_bow.columns.difference(['Sentiment'])]
    BOW_list = []
    i = 0
    for sent in df_2000['stemming_applied']:
        BOW_list.append(sent)
    nw = []
    for s in BOW_list:
        numOfWords = dict.fromkeys(wordset12, 0)
        for w in s:
            numOfWords[w] += 1
        nw.append(numOfWords)
    tf = []
    for b,n in zip(BOW_list,nw):
        tf.append(computeTF(n, b))
    idfs = computeIDF(nw)
    tfidf = []
    for tf in tf:
        tfidf.append(computeTFIDF(tf, idfs))

    df_tfidf = pd.DataFrame(tfidf)
    df_tfidf['Sentiment'] = df_2000["Sentiment"]
    training_data, testing_data = train_test_split(df_tfidf, test_size=0.2, random_state=25)
    x_train = training_data[training_data.columns.difference(['Sentiment'])]
    y_train = training_data['Sentiment'].values
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.preprocessing import StandardScaler

    # sc = StandardScaler()
    # sc.fit(x_train.values)
    bad_indices = np.where(np.isinf(x_train.values))
    print()
    # from sklearn.feature_extraction.text import CountVectorizer
    #
    # count_vector = CountVectorizer()
    #
    # # fit_transform() creates dictionary and return term-document matrix.
    # X_train_counts = count_vector.fit_transform(x_train.data)
    #
    # # Import TfidfTransformer class.
    # # TfidfTransformer transoforms count matrix to tf-idf representation.
    # from sklearn.feature_extraction.text import TfidfTransformer

    # tfidf_transformer = TfidfTransformer()

    # fit_transform transforms count matrix to tf-idf representation(vector).
    # X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(x_train, y_train)
    print()
    #
    # training_data, testing_data = train_test_split(df_bow, test_size=0.2, random_state=25)
    # x_train  = training_data[training_data.columns.difference(['Sentiment'])]
    # y_train = training_data['Sentiment'].values
    # t = []
    # for f, b in zip(x_train.to_dict(orient='records'), y_train):
    #     r = (f, b)
    #     t.append(r)
    #
    # x_val = testing_data[training_data.columns.difference(['Sentiment'])]
    # y_val = testing_data['Sentiment'].values
    # v = []
    # for f, b in zip(x_val.to_dict(orient='records'), y_val):
    #     r = (f, b)
    #     v.append(r)
    # #
    # # label_encoder = LabelEncoder()
    # #
    # # y_train = label_encoder.fit_transform(y_train)
    # # y_test = label_encoder.fit_transform(y_val)
    #
    # classifier = nltk.NaiveBayesClassifier.train(t);
    # print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, v)) * 100)
    # classifier = naive_bayes.MultinomialNB()
    # classifier.fit(x_train, y_train)
    #
    # predictions = classifier.predict(y_val)
    #
    # print(metrics.accuracy_score(predictions, y_test))









    print()

    # df_status_negative = pd.DataFrame(df[df.Sentiment == "Olumsuz"]).head(500)
    # df_status_positive = pd.DataFrame(df[df.Sentiment == "Olumlu"]).head(500)
    # df_500 = vertical_stack = pd.concat([df_status_positive, df_status_negative])
    #
    # print()
    # lamma_dict = {}
    # for lemma_list in df_500['stemming_applied']:
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
    # with open("lemmatization_500.json", "w") as outfile:
    #     json.dump(lamma_dict, outfile)

    # plot_drawing = PlotDrawing.PlotDraw()
    #
    #

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

    # with open('filename_train_dataset.pkl', 'rb') as inp:
    #     data_frame_pkl = pickle.load(inp)
    # # word_freq_dict = text_process.get_word_freq_dict(data_frame_pkl['stemming_applied'])
    # count = 0
    # for key in word_freq_dict.keys().__reversed__():
    #     if word_freq_dict[key] == 1:
    #         count += 1
    # df_status_negative = data_frame_pkl[data_frame_pkl.Durum == "Olumsuz"]
    # df_status_positive = data_frame_pkl[data_frame_pkl.Durum == "Olumlu"]
    # positive_bigram_list = []
    # negative_bigram_list = []
    # for sent in df_status_positive['stemming_applied']:
    #     bigram_list = []
    #     if len(sent) > 1:
    #         index = 0;
    #         while index +1 < len(sent):
    #             bigram_list.append([sent[index], sent[index + 1]])
    #             index += 1
    #     positive_bigram_list.append(bigram_list)
    # for sent in df_status_negative['stemming_applied']:
    #     bigram_list = []
    #     if len(sent) > 1:
    #         index = 0;
    #         while index +1 < len(sent):
    #             bigram_list.append([sent[index], sent[index + 1]])
    #             index += 1
    #     negative_bigram_list.append(bigram_list)
    # df_status_positive['bigram_lists'] = positive_bigram_list
    # df_status_negative['bigram_lists'] = negative_bigram_list
    # bigram_freq_dict_p = text_process.get_word_freq_dict(df_status_positive['bigram_lists'])
    # bigram_freq_dict_n = text_process.get_word_freq_dict(df_status_negative['bigram_lists'])
    # print()
    # plot_drawing.DrawBarPlot(bigram_freq_dict_p)
    # word_freq_dict_positive  = text_process.get_word_freq_dict(df_status_positive['stemming_applied'])
    # word_freq_dict_negative  = text_process.get_word_freq_dict(df_status_negative['stemming_applied'])
    # # plot_drawing.DrawBarPlot(word_freq_dict_negative)
    # d = {}
    # d['Positive'] = word_freq_dict_positive['kötü']
    # d['Negative'] = word_freq_dict_negative['kötü']
    # font = {'family': 'normal',
    #         'weight': 'bold',
    #         'size': 16}
    # matplotlib.rc('font', **font)
    # courses = list(d.keys())
    # values = list(d.values())
    #
    # fig = plt.figure(figsize=(4, 3))
    #
    # # creating the bar plot
    # plt.bar(courses, values, color='black',
    #         width=0.75)
    #
    # plt.xlabel("Sentiment", fontsize=18)
    # plt.ylabel("kötü", fontsize=18)
    # plt.title("")
    # plt.show()

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
