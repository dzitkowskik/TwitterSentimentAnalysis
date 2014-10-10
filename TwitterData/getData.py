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


@classmethod
def parse(cls, api, raw):
    status = cls.first_parse(api, raw)
    setattr(status, 'json', json.dumps(raw))
    return status


def get_config(name):
    f = file(name)
    return Config(f)


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


def get_time_left_str(cur_idx, fetch_list, download_pause):
    tweets_left = len(fetch_list) - cur_idx
    total_seconds = tweets_left * download_pause
    str_hr = int(total_seconds / 3600)
    str_min = int((total_seconds - str_hr*3600) / 60)
    str_sec = total_seconds - str_hr*3600 - str_min*60
    return '%dh %dm %ds' % (str_hr, str_min, str_sec)


def download_tweets(fetch_list, raw_dir, tweeter_api, db):
    # ensure raw data directory exists
    if not os.path.exists(raw_dir):
        os.mkdir(raw_dir)

    # stay within rate limits
    max_tweets_per_hr = 700
    download_pause_sec = 3600 / max_tweets_per_hr

    # download tweets
    for idx in range(0, len(fetch_list)):
        # current item
        item = fetch_list[idx]

        # print status
        trem = get_time_left_str(idx, fetch_list, download_pause_sec)
        print '--> downloading tweet #%s (%d of %d) (%s left)' % \
              (item[2], idx+1, len(fetch_list), trem)

        try:
            status = tweeter_api.get_status(id=item[2])
        except tweepy.error.TweepError, e:
            if 'Rate limit exceeded' not in e.message:
                print item[2], e.message
        else:
            if db.test_tweets.find({'_id': item[2]}).count() == 0:          
                # download to db
                record = [{
                    '_id': item[2],
                    'topic': item[0],
                    'manual_grade': item[1],
                    'text': status.text,
                    'retweet_count': status.retweet_count
                    }]
                db.test_tweets.insert(record)

                # stay in Twitter API rate limits
                print '    pausing %d sec to obey Twitter API rate limits' % \
                      (download_pause_sec)
                time.sleep(download_pause_sec)
            else:
                print 'Tweet ', item[2], ' already downloaded!\n'
    return


def main():

    # get configuration
    cfg = get_config('configuration.cfg')

    # get database connection
    db_client = MongoClient(cfg.db_host, int(cfg.db_port))

    # get database
    db = db_client[cfg.db_database_name]

    # get tweeter api
    api = get_tweepy_api(cfg)

    # fetch list and download tweets
    print "Getting tweet list...\n"
    tweet_list = read_total_list(cfg.corpus_file)
    print "Downloading tweets...\n"
    download_tweets(tweet_list, cfg.raw_dir, api, db)

    # disconnect database
    client.close()

    # second pass for any failed downloads
    print '\nDownloading tweets done!\n'

    return


if __name__ == '__main__':
    main()
