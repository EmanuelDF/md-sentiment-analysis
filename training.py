import re
import csv
import string

from textblob.classifiers import NaiveBayesClassifier


def clean_tweet(tweet):
    tweet = str(tweet).replace(';', '\',\'').lower().strip()
    tweet = re.sub(r"[\([{})\]]", "", tweet)
    tweet = re.sub(r'([a-z])\1+', r'\1', tweet)
    tweet = re.compile('<.*?>').sub('', tweet)
    tweet = re.sub('\s+', ' ', tweet)
    tweet = re.sub(r'\[[0-9]*\]', ' ', tweet)
    tweet = re.sub(r'\d', ' ', tweet)
    tweet = re.sub(r'\s+', ' ', tweet)
    return tweet


if __name__ == '__main__':
    array_model = []
    file = open('resource/model.csv', 'r')
    try:
        reader = csv.reader(file)
        for each_row in reader:
            array_model.append(clean_tweet(each_row))
    finally:
        file.close()

    customized_classifier = NaiveBayesClassifier(array_model)
