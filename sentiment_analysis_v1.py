mport os
import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob
import matplotlib.pyplot as plt

query = 'nome social -filter:retweets'
count = 100

class TwitterClient(object):

    def __init__(self):
        consumer_key = os.environ['CONSUMER_KEY']
        consumer_secret = os.environ['CONSUMER_SECRET']
        access_token = os.environ['ACCESS_TOKEN']
        access_token_secret = os.environ['ACESS_TOKEN_SECRET']

        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'

    def get_tweets(self, query, count):
        tweets = []
        try:
            fetched_tweets = self.api.search_tweets(q=query, count=count)
            for tweet in fetched_tweets:
                parsed_tweet = {}
                parsed_tweet['text'] = tweet.text
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
                if tweet.retweet_count > 0:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            return tweets

        except AttributeError as e:
            print("Error : " + str(e))


def main():
    api = TwitterClient()
    tweets = api.get_tweets(query, count=count)

    ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
    print("Positives tweets: {} %".format(100 * len(ptweets) / len(tweets)))
    ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
    print("Negatives tweets: {} %".format(100 * len(ntweets) / len(tweets)))
    print("Neutral tweets: {} %".format(100 * (len(tweets) - len(ntweets) - len(ptweets)) / len(tweets)))

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
    print("Pie-Chart representation:\n")
    plt.show()

if __name__ == "__main__":
    main()