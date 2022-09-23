import os
import re
import tweepy
import numpy as np
import matplotlib.pyplot as plt
from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier

max_results = 10
query = '"nome social" place_country:BR lang:pt -is:retweet'

class TwitterClient(object):

    def __init__(self):
        consumer_key = os.environ['CONSUMER_KEY']
        consumer_secret = os.environ['CONSUMER_SECRET']
        bearer_token = os.environ['BEARER_TOKEN']
        access_token = os.environ['ACCESS_TOKEN']
        access_token_secret = os.environ['ACCESS_TOKEN_SECRET']

        try:
            self.api = tweepy.Client(bearer_token=bearer_token, consumer_key=consumer_key,
                                     consumer_secret=consumer_secret, access_token=access_token,
                                     access_token_secret=access_token_secret)
        except:
            print('Error: Authentication Failed')

    def clean_tweet(self, tweet):
        return ' '.join(re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)', ' ', tweet).split())

    def get_tweet_sentiment(self, tweet, customized_classifier):
        analysis = TextBlob(self.clean_tweet(tweet), classifier=customized_classifier)

        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, max_results):
        customized_classifier = self.train()
        tweets = []
        try:
            fetched_tweets = self.api.search_all_tweets(
                query=query, tweet_fields=['context_annotations', 'created_at', 'geo'],
                place_fields=['place_type', 'geo'], expansions='geo.place_id', max_results=max_results)
            for tweet in fetched_tweets.data:
                parsed_tweet = {}
                parsed_tweet['text'] = tweet.text
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(
                    tweet.text, customized_classifier)
                tweets.append(parsed_tweet)

            return tweets

        except AttributeError as e:
            print('Error : ' + str(e))

    def train(self):
        train = [
            ('Ainda usam o meu dead name!', 'neg'),
            ('Estou muito orgulhosa com meu nome social.', 'pos'),
            ('Nome social? Pessoas acham que é besteira!', 'neg'),
            ('Bem vinda, vida nova! #nomesocial', 'pos'),
            ('Finalmente um site q deixa usar nome social.', 'pos'),
            ('Ignoram meu nome social', 'neg')
        ]
        return NaiveBayesClassifier(train)

def main():
    api = TwitterClient()
    tweets = api.get_tweets(query, max_results=max_results)

    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    print("Positives tweets: {} %".format(100 * len(ptweets) / len(tweets)))
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    print("Negatives tweets: {} %".format(100 * len(ntweets) / len(tweets)))
    print("Neutral tweets: {} %".format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets)))

    print(np.array(tweets))

    ptweetspie = 100 * len(ptweets) / len(tweets)
    ntweetspie = 100 * len(ntweets) / len(tweets)
    neutraltweetspie = 100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets)
    sizes = [ptweetspie, ntweetspie, neutraltweetspie]
    labels = 'Positives tweets', 'Negatives tweets', 'Neutral tweets'

    explode = (0, 0.1, 0.1)
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')
    print('Pie-Chart representation:\n')
    plt.show()

if __name__ == '__main__':
    main()