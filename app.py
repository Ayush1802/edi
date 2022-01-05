import os
import sys
from contextlib import contextmanager
# twitter
from datetime import datetime
from io import StringIO
from threading import current_thread
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from lxml import html
from pandas_datareader import data as web
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from datetime import date
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#start = datetime(2021, 1, 1)


today = date.today()
#start_date = st.date_input('Start date', today)
#end_date = st.date_input('End date', tomorrow)

start = st.date_input('Start date', today)
end = datetime(2021, 12,10)



@contextmanager
def st_redirect(src, dst):
    placeholder = st.empty()
    output_func = getattr(placeholder, dst)

    with StringIO() as buffer:
        old_write = src.write

        def new_write(b):
            if getattr(current_thread(), REPORT_CONTEXT_ATTR_NAME, None):
                buffer.write(b)
                output_func(buffer.getvalue())
            else:
                old_write(b)

        try:
            src.write = new_write
            yield
        finally:
            src.write = old_write


@contextmanager
def st_stdout(dst):
    with st_redirect(sys.stdout, dst):
        yield


@contextmanager
def st_stderr(dst):
    with st_redirect(sys.stderr, dst):
        yield

option = st.sidebar.selectbox("Which Dashboard?", ('twitter', 'wallstreetbets', 'stocktwits', 'chart', 'pattern'), 3)



st.title('Stock Trend Predication')
user_input = st.text_input('Enter stock ticker','AAPL')
df = web.get_data_yahoo(user_input,start=start,end=end)
#df = web.DataReader(user_input,'yahoo',start=start,end=end)
#dff = web.get_balance_sheet(user_input,start=start,end=end)

st.subheader('data from 2010-2021')
st.write(df.describe())

st.subheader('Closing price vs time chart')
fig = plt.figure(figsize =(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing price vs time chart 100 Ma')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize =(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)



data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

# x_train = []
# y_train = []
#
# for i in range(100, data_training_array.shape[0]):
#     x_train.append(data_training_array[i - 100:i])
#     y_train.append(data_training_array[i, 0])
#
# x_train , y_train = np.array(x_train), np.array(y_train)

from keras import models
# model = models.load_model('filename.h5')

model = models.load_model('keras_model.h5')
past_100_days = data_training.tail(100)
final_df=past_100_days.append(data_testing,ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i,0])

x_test ,y_test = np.array(x_test) , np.array(y_test)
y_predicted = model.predict(x_test)
scaler = scaler.scale_


scale_factor = 1/scaler[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

st.subheader('Fundamental')
fig2 = plt.figure(figsize =(12,6))
plt.plot(y_test,'b',label = 'Orginial price')
plt.plot(y_predicted,'r',label = 'Pred price')
st.pyplot(fig2)




msft = yf.Ticker(user_input)
msft.history(period="1y")

# show actions (dividends, splits)
msft.recommendations



symbol = user_input

url = 'https://finance.yahoo.com/quote/' + symbol + '/balance-sheet?p=' + symbol

#
headers = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
    'Cache-Control': 'max-age=0',
    'Connection': 'close',
    'DNT': '1', # Do Not Track Request Header
    'Pragma': 'no-cache',
    'Referrer': 'https://google.com',
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
}

page = requests.get(url, headers=headers)

tree = html.fromstring(page.content)

tree.xpath("//h1/text()")

table_rows = tree.xpath("//div[contains(@class, 'D(tbr)')]")

assert len(table_rows) > 0

parsed_rows = []

for table_row in table_rows:
    parsed_row = []
    el = table_row.xpath("./div")

    none_count = 0

    for rs in el:
        try:
            (text,) = rs.xpath('.//span/text()[1]')
            parsed_row.append(text)
        except ValueError:
            parsed_row.append(np.NaN)
            none_count += 1

    if (none_count < 4):
        parsed_rows.append(parsed_row)

df = pd.DataFrame(parsed_rows)


df = pd.DataFrame(parsed_rows)
df = df.set_index(0) # Set the index to the first column Period Ending.
df = df.transpose()

cols = list(df.columns)
cols[0] = 'Date'
df = df.set_axis(cols, axis='columns', inplace=False)

df

st.title('Tweet Sentiment Analysis')




import re
import tweepy
from tweepy import OAuthHandler
from textblob import TextBlob


class TwitterClient(object):

    def __init__(self):

        consumer_key = 'Py0nJPLCN0F7Z3WNHtPCKgAxl'
        consumer_secret = 'uMp5Ls2FzNiwPsImGMyPZARwzaCxIcsPGtClYsyUvC2xqnJdFD'
        access_token = '3553689916-HEuAPbMQxVOCPsQqHmyZgv8GJKKNH0MmLxH5uyh'
        access_token_secret = 'aIrPqZVLzV1sabwRhNzHUqCnlyEkPEAbsoOSxzHjgltUm'

        # authentication
        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_token_secret)
            # create tweepy API object to fetch tweets
            self.api = tweepy.API(self.auth)
        except:
            print("Error: Authentication Failed")

    def clean_tweet(self, tweet):
        '''
        Utility function to clean tweet text by removing links, special characters using simple regex statements.
        '''
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

    def get_tweet_sentiment(self, tweet):
        '''
        Utility function to classify sentiment of passed tweet
        using textblob's sentiment method
        '''
        # create TextBlob object of passed tweet text
        analysis = TextBlob(self.clean_tweet(tweet))
        # set sentiment
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity == 0:
            return 'neutral'
        else:
            return 'negative'


    def get_tweets(self, query, count=10):
        '''
        Main function to fetch tweets and parse them.
        '''
        # empty list to store parsed tweets
        tweets = []

        try:
            # call twitter api to fetch tweets
            # fetched_tweets = self.api.search(q = query, count = count)
            fetched_tweets = self.api.search_tweets(q=query, count=count)
            S = []

            # parsing tweets one by one
            for tweet in fetched_tweets:
                # empty dictionary to store required params of a tweet
                parsed_tweet = {}

                # saving text of tweet
                parsed_tweet['text'] = tweet.text
                # saving sentiment of tweet
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
                analysis = TextBlob(tweet.text)
                S.append(analysis.sentiment)
                # appending parsed tweet to tweets list
                if tweet.retweet_count > 0:
                    # if tweet has retweets, ensure that it is appended only once
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)

            polar = [i[0] for i in S]
            subject = [i[1] for i in S]
            # return parsed tweets
            return tweets, polar, subject

        except tweepy.TweepError as e:
            # print error (if any)
            print("Error : " + str(e))


#
# def main():
#     # creating object of TwitterClient Class
#     api = TwitterClient()
#     #q = input("Enter the stock to be analyzed : ")
#     q="asian paint"
#     user = st.text_input('Enter the stock to be analyzed : ')
#     q=user
#     tweets = api.get_tweets(query = q, count = 10)
#
#     # printing first 10 tweets
#     print("\n\nRecent Tweets:")
#     with st_stdout("info"):
#         for tweet in tweets:
#             print("UNCLEAN : ",tweet['text'])
#             print("CLEAN : ",tweet['clean'],"\n")


def main():
    # creating object of TwitterClient Class
    api = TwitterClient()

    q = "asian paint"
    user = st.text_input('Enter the stock to be analyzed : ',value='asian paint', type='default')
    q=user
    #tweets, polar, subject = api.get_tweets(query=q, count=100)
    tweets, polar, subject = api.get_tweets(query=q, count=100)

    with st_stdout("info"):
        # picking positive tweets from tweets
        ptweets = [tweet for tweet in tweets if tweet['sentiment'] == 'positive']
        # percentage of positive tweets
        print("Positive tweets percentage: {} %".format(100 * len(ptweets) / len(tweets)))
        # picking negative tweets from tweets
        ntweets = [tweet for tweet in tweets if tweet['sentiment'] == 'negative']
        # percentage of negative tweets
        print("Negative tweets percentage: {} %".format(100 * len(ntweets) / len(tweets)))
        # percentage of neutral tweets
        print("Neutral tweets percentage: {} % \
        ".format(100 * (len(tweets) - (len(ntweets) + len(ptweets))) / len(tweets)))

        # printing first 5 positive tweets
        print("\n\nPositive tweets:")
        for tweet in ptweets[:10]:
            print(tweet['text'])

        # printing first 5 negative tweets
        print("\n\nNegative tweets:")
        for tweet in ntweets[:10]:
            print(tweet['text'])

    labels = 'POSTIVE','NEGATIVE','NETURAL'
    sizes = [100 * len(ptweets) / len(tweets),100 * len(ntweets) / len(tweets), 100 * (len(tweets) - (len(ntweets) + len(ptweets))) / len(tweets)]
    explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes,labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.pyplot(fig1)

    print(polar)
    print(subject)



    ax = sns.kdeplot(polar, shade=True, color="g")
    ax = sns.kdeplot(subject, shade=True, color="b")


if __name__ == "__main__":
    # calling main function
    main()
#st.write(tweet)




#streamlit run app.py