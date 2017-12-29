# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 16:22:07 2017

@author: Ashish Kumar Jayant
@title: Tweet Sentiment Analyser (mainly for reviews,complaints and feedbacks,
                                  not so good for political conversations)

"""
#import tweepy
from textblob import TextBlob
#import csv
import pandas as pd
#from tweepy import OAuthHandler
import re
from twython import Twython
import nltk
#import pickle
import sys
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from twython import TwythonStreamer
import sqlite3
from sqlite3 import Error



#------------BY HASH TAG----

class TweetStreamer(TwythonStreamer):
    c=1
    def on_success(self, data):
        if 'text' in data:
            print(data['text'].encode('utf-8'),"sentiment_1 = ",return_tweet_sentiment(str(data['text'].encode('utf-8'))),"sentiment_2 - ",test_sentiment(str(data['text'])), "high confidence" if test_sentiment(str(data['text']))==return_tweet_sentiment(str(data['text'].encode('utf-8'))) else "low confidence")
            d = (tag,data['text'].encode('utf-8'),return_tweet_sentiment(str(data['text'].encode('utf-8'))),test_sentiment(str(data['text'])))
            insert_row(conn,d,table)
            TweetStreamer.c+=1
            if TweetStreamer.c>5:
                self.disconnect()
    def on_error(self, status_code, data):
        if status_code==420:
            print("Too many requests in present window!")
        self.disconnect()

def get_tweets_hashtag(tag):
    t = Twython(app_key=consumer_key, 
            app_secret=consumer_secret, 
            oauth_token=access_token, 
            oauth_token_secret=access_secret)
    search = t.search(q=tag,count=100,lang="en",result_type="recent")#result_type="popular")
    tweets = search['statuses']
    #print(tweets[0])
    tweet_text = [i['text'] for i in tweets]
    tweet_location = [i['user']['location'] for i in tweets if i['retweet_count']==0]
    tweet_text = [tweet.encode("utf-8") for tweet in tweet_text]
    tweet_location = [tweet.encode("utf-8") for tweet in tweet_location]
    df = pd.DataFrame({"tweet_text": pd.Series(tweet_text),"location":pd.Series(tweet_location)})
    #df.drop_duplicates(inplace=True)
    print("Analysing :",df.shape[0]," tweets..........")
    df.to_csv("tag_tweet_data.csv")

    
''' #BY USER HANDLE------------------
def get_all_tweets(screen_name):
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
	
    #initialize a list to hold all the tweepy Tweets
    alltweets = []	
    #make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.user_timeline(screen_name = screen_name,count=200)
	#save most recent tweets
    alltweets.extend(new_tweets)
    #save the id of the oldest tweet less one
    oldest = alltweets[-1].id - 1
    #keep grabbing tweets until there are no tweets left to grab
    while len(new_tweets) > 0:
        print("getting tweets before %s" % (oldest))
		#all subsiquent requests use the max_id param to prevent duplicates
        new_tweets = api.user_timeline(screen_name = screen_name,count=200,max_id=oldest)
		#save most recent tweets
        alltweets.extend(new_tweets)
		#update the id of the oldest tweet less one
        oldest = alltweets[-1].id - 1
        print ("...%s tweets downloaded so far" % (len(alltweets)))
        if len(alltweets)>350:
            break
	
	#transform the tweepy tweets into a 2D array that will populate the csv	
    outtweets = [[tweet.id_str, tweet.created_at, tweet.text.encode("utf-8")] for tweet in alltweets]
	
	#write the csv	
    with open('%s_tweets.csv' % screen_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["id","created_at","text"])
        writer.writerows(outtweets)
	
    pass
'''
def make_feature(size,txt):#,browser,device):
        import numpy as np
        import nltk
        x=np.zeros([size])
        st =  nltk.stem.SnowballStemmer('english')
        stops = set(nltk.corpus.stopwords.words("english"))
        txt = " ".join([w for w in txt.split(" ") if w not in stops])
        txt = " ".join([w.lower() for w in txt.split(" ")])
        txt = " ".join([st.stem(w) for w in txt.split(" ")]) 
        x[0]=2
        x[1]=1
        x[2]=len(txt)
        fit_new = CountVectorizer(analyzer='word',ngram_range=(1,3),min_df=1,max_features=1000)        
        fit_data= fit_new.fit_transform(pd.DataFrame({txt: pd.Series([txt])}))
        txt_vocab = fit_new.get_feature_names()
        vocab=count_fit.get_feature_names()
        for i in txt_vocab:
            try:
                ind =  vocab.index(i)
                if x[ind+3] == 0:
                    x[ind+3] =1
                else:
                    x[ind+3] +=1
            except:
                pass
        x=x.reshape(1,-1)
        return x    
    
    
def to_labels(x):
    if abs(x[0][0] - x[0][1])>0.15:
        if x[0][0]<x[0][1]: 
            return "positive"
        else:
            return "negative"
    else:
        return "neutral"

        
def test_sentiment(txt):
    x=make_feature(feats,txt)
    return to_labels(eclf.predict_proba(x))    

def load_pickle():
    files = ["my_ensemble.pkl","feats.pkl","vectorizer.pkl"]
    eclf = joblib.load(files[0])
    feats = joblib.load(files[1])
    count_fit = joblib.load(files[2])        
    print("Models loaded")
    return eclf,feats,count_fit


def clean_text(text):
    txt = str(text)
    pattern = re.compile('.http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))*')
    txt = pattern.sub("",txt)
    txt = re.sub(r'x(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))*',r'',txt)
    txt = re.sub(r'\n',r' ',txt)
    txt = re.sub(r'[^A-Za-z0-9\s]',r' ',txt)
    #words = nltk.corpus.words.words()
    #txt = " ".join([w for w in txt.split(" ") if w in words])
    return txt

def return_sentiment_csv():
    c = pd.read_csv(csv_file)
    c['tweet_text'] = c['tweet_text'].map(lambda x:clean_text(x))    
    #--------------textblob model-------------------------------
    c['score'] = c['tweet_text'].map(lambda x: TextBlob(x).sentiment.polarity)
    c['sentiment1'] = c['score'].map(lambda x: 'good' if x>0 else ('neutral' if x==0 else 'bad'))    
    #---------------my model - ensemble of random_forest,naive_bayes,gradient_boost,ada_boost trained on reviews data---------   
    c['sentiment2'] = c['tweet_text'].map(lambda x: test_sentiment(x))
    #giving confidence score
    match = (c['sentiment1'] == c['sentiment2'])
    c['confidence'] = match.map(lambda x: 'high' if x==True else 'low')
    return c

def insights_csv(tag):
    get_tweets_hashtag(tag);
    c = return_sentiment();
    print("---------------------------------------------------")
    print("Sentiment Insights for ",tag)
    print("---------------------------------------------------")
    h_c=c.loc[c.confidence=='high']
    h_c['sentiment1'].value_counts().plot(kind="bar",title=("Sentiment for %s"%tag))
    plt.show()
    #c['sentiment2'].value_counts().plot(kind="bar",title=("Sentiment for %s"%tag))
    file_name=("%s.csv"%tag)
    c.to_csv(file_name,sep="^")
    file_name_hc=("%s_high_confidence.csv"%tag)
    h_c.to_csv(file_name_hc,sep="^")
    print("---------------------------------------------------")
    print("Top positive tweets")
    pd.options.display.max_colwidth=1000
    x=h_c.sort_values('score',ascending=False).head(3)
    print(x[["tweet_text"]])
    print("---------------------------------------------------")
    print("Top negative tweets")
    x=h_c.sort_values('score',ascending=True).head(3)
    print(x[["tweet_text"]])
    return c

def return_tweet_sentiment(tweet):
    tweet = clean_text(tweet)
    pol=TextBlob(tweet).sentiment.polarity
    if pol==0:
        return 'neutral'
    elif pol>0:
        return 'positive'
    else:
        return 'negative'
    
    
def stream_tweet_sentiments(tag):    
    streamer = TweetStreamer(consumer_key, consumer_secret,
                         access_token, access_secret)

    streamer.statuses.filter(track=tag)

def create_connection(db_name):
    try:
        conn = sqlite3.connect(db_name)
        print(sqlite3.version)
        return conn
    except Error as e:
        print(e)
    return None


        

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)        
    except Error as e:
        print(e)            


def insert_row(conn,data,table):
    sql = ''' INSERT INTO %s'''%table+'''(tag,tweet_text,sentiment1,sentiment2)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql,data)
    print(" --> inserted into db (done)")
    conn.commit()
     
#-----DRIVER---------------------

consumer_key = 'XXXXXXXXXXXXXXXXXXXXXX'
consumer_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_token = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
access_secret = 'XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX'
eclf,feats,count_fit = load_pickle()
csv_file="tag_tweet_data.csv" 
tag=input("Enter hashtag to be analysed - ")
conn=create_connection("my_db")
table="my_table"
create_table_sql = """ CREATE TABLE IF NOT EXISTS my_table ( tag text, tweet_text text, sentiment1 text, sentiment2 text );"""
create_table(conn,create_table_sql)
stream_tweet_sentiments(tag)
conn.close()
