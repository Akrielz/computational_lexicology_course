import os

import pandas as pd
import tweepy
from dotenv import load_dotenv
from tqdm import tqdm


def main(path):
    # load the ".env" file
    load_dotenv()

    # get the twitter api keys
    consumer_key = os.getenv("twitter_api_key")
    consumer_secret = os.getenv("twitter_api_secret_key")
    access_token = os.getenv("twitter_access_token")
    access_token_secret = os.getenv("twitter_access_token_secret")

    # create the twitter api object
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    api = tweepy.API(auth)

    # open a dataframe with pandas
    df = pd.read_csv(path, sep="\t")
    df = df[df["Expert"] == "sexism"]

    # get the twitter_ids
    twitter_ids = df["TweetID"].unique()

    # grab the tweets from the twitter_ids
    tweets = []
    for twitter_id in tqdm(twitter_ids):
        try:
            tweet = api.get_status(twitter_id).text
            tweets.append(tweet)
        except Exception as e:
            continue

    # create a new dataframe with the tweets
    df_new = pd.DataFrame(tweets, columns=["text"])

    # save the dataframe as csv
    path_csv = path.replace(".csv", "_downloaded.csv")
    df_new.to_csv(path_csv, index=False)


if __name__ == "__main__":
    load_dotenv()
    # main("../data/compliments/benevolent_sexist.tsv")
    # main("../data/compliments/hostile_sexist.tsv")
    # main("../data/hate_speech/NAACL_SRW_2016.csv")
    main("../data/hate_speech/NLP+CSS_2016.csv")
