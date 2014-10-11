# Twitter-Data Sentiment Get Data Script based on
# Sanders-Twitter Sentiment Corpus Install Script - Niek Sanders
# By
# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014

import csv
import getpass
import json
import os
import time
import urllib
import tweepy
from tweepy import Cursor
from config import Config
from pymongo import MongoClient
import json
import wordSentiment


@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status


def get_wait_time(cfg):
    return 3600.0 / int(cfg.max_tweets_per_hour)


def get_tweepy_api(cfg):
    tweepy.models.Status.first_parse = tweepy.models.Status.parse
    tweepy.models.Status.parse = parse

    # == OAuth Authentication ==
    auth = tweepy.OAuthHandler(cfg.consumer_key, cfg.consumer_secret)
    auth.secure = True
    auth.set_access_token(cfg.access_token, cfg.access_token_secret)

    # Get tweepy api
    return tweepy.API(auth)


def read_total_list(in_filename):
    # read total fetch list csv
    fp = open(in_filename, 'rb')
    reader = csv.reader(fp, delimiter=',', quotechar='"')

    total_list = []
    for row in reader:
        total_list.append(row)

    return total_list


def get_time_left_str(cur_idx, length, download_pause):
    tweets_left = length - cur_idx
    total_seconds = tweets_left * download_pause
    str_hr = int(total_seconds / 3600)
    str_min = int((total_seconds - str_hr*3600) / 60)
    str_sec = total_seconds - str_hr*3600 - str_min*60
    return '%dh %dm %ds' % (str_hr, str_min, str_sec)


def save_tweet(db, item, status, active, analyzer):
    if active:
        word_sentiment = analyzer.get_word_sentiment(status.text)
        record = [{
            '_id': item[2],
            'isActive': True,
            'topic': item[0],
            'manual_grade': item[1],
            'word_sentiment': word_sentiment,
            'text': status.text,
            'retweet_count': status.retweet_count,
            'data': json.loads(status.json)
            }]
    else:
        record = [{'_id': item[2], 'isActive': False}]
    db.test_tweets.insert(record)


def wait_between_requests(idx, length, download_pause_sec):
    # stay in Twitter API rate limits
    trem = get_time_left_str(idx, length, download_pause_sec)
    print '    pausing %f sec to obey Twitter API rate limits (%s left)' % \
          (download_pause_sec, trem)
    time.sleep(download_pause_sec)


def download_tweets(fetch_list, tweeter_api, db, download_pause_sec, analyzer):
    length = len(fetch_list)
    for idx in range(0, length):
        item = fetch_list[idx]
        if db.test_tweets.find({'_id': item[2]}).count() == 0:
            try:
                print '--> downloading tweet #%s (%d of %d)' % \
                    (item[2], idx+1, length)
                status = tweeter_api.get_status(id=item[2])
            except tweepy.error.TweepError, e:
                print 'ERROR - %s (Tweet: %s)' % \
                    (e.message[0]['message'], item[2])
                save_tweet(db, item, None, False, None)
            else:
                save_tweet(db, item, status, True, analyzer)
            finally:
                wait_between_requests(idx, length, download_pause_sec)
        else:
            print 'Tweet ', item[2], ' already downloaded!'
    return


def main():
    print "Downloading tweets...\n"

    # get configuration
    config_file_name = 'configuration.cfg'
    cfg = Config(file(config_file_name))

    # get database
    db_client = MongoClient(cfg.db_host, int(cfg.db_port))
    db = db_client[cfg.db_database_name]

    # get tweeter api
    api = get_tweepy_api(cfg)

    # initialize word sentiment analyzer
    analyzer = wordSentiment.WordSentimentAnalyzer(config_file_name)

    # fetch list and download tweets
    tweet_list = read_total_list(cfg.corpus_file)
    download_tweets(tweet_list, api, db, get_wait_time(cfg), analyzer)

    # disconnect from database
    client.close()

    print '\nDownloading tweets done!\n'
    return


if __name__ == '__main__':
    main()
