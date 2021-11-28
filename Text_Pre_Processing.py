import re

import nltk
from click import secho
from nltk.lm.preprocessing import pad_both_ends
from nltk.util import bigrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from snowballstemmer import stemmer
import zeyrek


class TextPreProcessing(object):

    def __init__(self):
        print("Text_Processing Class")

    def lower_case_apply(self, sentences_lst):
        lower_case_sent_list = []
        for s in sentences_lst:
            lower_case_sent_list.append(s.lower())
        return lower_case_sent_list

    def unique_vowels_change(self, sentences_lst):
        vowel_changed_sentences = []
        for sent in sentences_lst:

            sent = re.sub('ü', 'u', sent)
            sent = re.sub('ı', 'i', sent)
            sent = re.sub('ş', 's', sent)
            sent = re.sub('ö', 'o', sent)
            sent = re.sub('ç', 'c', sent)
            sent = re.sub('ğ', 'g', sent)
            vowel_changed_sentences.append(sent)
        return vowel_changed_sentences

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

    def stemming(self, sent_lst):
        stemming_applied_lst = []
        stem_obj = stemmer('turkish')
        analyzer = zeyrek.MorphAnalyzer()
        print(analyzer.lemmatize('Ürün'))
        for sent in sent_lst:
            print(sent)
            lemma_list = []
            for token in sent:
                lemma_list.append(analyzer.lemmatize(token))
            print(lemma_list)
            stemming_applied_lst.append(lemma_list)

        return stemming_applied_lst