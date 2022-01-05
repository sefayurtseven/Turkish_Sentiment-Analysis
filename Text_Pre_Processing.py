import re

import nltk
from click import secho
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from snowballstemmer import stemmer
import zeyrek
from collections import OrderedDict


class TextPreProcessing(object):

    def __init__(self):
        print("Text_Processing Class")

    def lower_case_apply(self, sentences_lst):
        lower_case_sent_list = []
        for s in sentences_lst:
            lower_case_sent_list.append(s.lower())
        return lower_case_sent_list

    def remove_punctuations(self, sent_lst):
        punctuations_removed_sentences = []
        for sent in sent_lst:
            punc = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
            for ele in sent:
                if ele in punc:
                    sent = sent.replace(ele, " ")
            punctuations_removed_sentences.append(sent)
        return punctuations_removed_sentences

    def tokenize_sentences(self, sentences_lst):
        tokenized_sent_lst = []
        for sent in sentences_lst:
            tokenized_sent_lst.append(nltk.word_tokenize(sent))
        return tokenized_sent_lst

    def remove_stop_words(self, sent_lst, stop_words_list):
        stop_words_removed_lst = []
        removed_stop_word_count = 0
        for sent in sent_lst:
            for sw in stop_words_list:
                if list(sent).__contains__(sw):
                    sent.remove(sw)
                    removed_stop_word_count += 1
            stop_words_removed_lst.append(sent)
        return stop_words_removed_lst

    def sent_lemmatize(self, sent_lst):
        stemming_applied_lst = []
        analyzer = zeyrek.MorphAnalyzer()
        for sent in sent_lst:
            lemma_list = []
            for token in sent:
                print(analyzer.lemmatize(token))
                if len(analyzer.lemmatize(token)) > 0:
                    lemma_list.append(analyzer.lemmatize(token)[0][1][0])
            stemming_applied_lst.append(lemma_list)

        return stemming_applied_lst

    def get_word_freq_dict(self, sent_lst):
        word_freq_dict = {}
        for s in sent_lst:
            for w in s:
                if word_freq_dict.keys().__contains__(w):
                    word_freq_dict[w] = word_freq_dict[w] + 1
                else:
                    word_freq_dict[w] = 1

        word_freq_dict = {k: v for k, v in sorted(word_freq_dict.items(), key=lambda item: item[1])}
        res = OrderedDict(reversed(list(word_freq_dict.items())))
        return res

    def get_word_freq_dict2(self, sent_lst):
        word_freq_dict = {}
        for s in sent_lst:
            print(s)
            for w in s:
                if len(w) >= 1:
                    if word_freq_dict.keys().__contains__(w[0] + ", " + w[1]):
                        word_freq_dict[w[0] + ", " + w[1]] = word_freq_dict[w[0] + ", " + w[1]] + 1
                    else:
                        word_freq_dict[w[0]+ ", " + w[1]] = 1

        word_freq_dict = {k: v for k, v in sorted(word_freq_dict.items(), key=lambda item: item[1])}
        res = OrderedDict(reversed(list(word_freq_dict.items())))
        return res