# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import csv
import time
import json
import core
import tweepy
from tweepy import error
from config import Config
import inject
from pymongo import MongoClient
from wordSentiment import WordSentimentAnalyzer


class TweetDownloader(object):
    @inject.params(config=Config, db_client=MongoClient)
    def __init__(self, config, db_client):
        self.cfg = config
        self.db = db_client[config.db_database_name]
        self.tweeter_api = core.get_tweepy_api(self.cfg)

    def _get_wait_time(self):
        return 3600.0 / int(self.cfg.max_tweets_per_hour)

    @staticmethod
    def __read_tweet_list(filename):
        # read total fetch list csv
        fp = open(filename, 'rb')
        reader = csv.reader(fp, delimiter=',', quotechar='"')

        total_list = []
        for row in reader:
            total_list.append(row)

        return total_list

    @staticmethod
    def __get_time_left_str(cur_idx, length, download_pause):
        tweets_left = length - cur_idx
        total_seconds = tweets_left * download_pause
        str_hr = int(total_seconds / 3600)
        str_min = int((total_seconds - str_hr * 3600) / 60)
        str_sec = total_seconds - str_hr * 3600 - str_min * 60
        return '%dh %dm %ds' % (str_hr, str_min, str_sec)

    @staticmethod
    def undersample(sentiment, manual_grade, threshold=0.25):
        if abs(sentiment) < threshold and manual_grade is None:
            return True
        else:
            return False

    @staticmethod
    def _save_tweet(table, status, active, analyzer, tag, item=None, undersample=False):
        if status is not None:
            word_sentiment = analyzer.get_word_sentiment(status.text)
            topic = "" if item is None else item[0]
            manual_grade = None if item is None else item[1]

            if undersample:
                if TweetDownloader.undersample(word_sentiment, manual_grade):
                    return

            record = {
                '_id': status.id_str,
                'isActive': active,
                'tag': tag,
                'topic': topic,
                'manual_grade': manual_grade,
                'word_sentiment': word_sentiment,
                'text': status.text,
                'retweet_count': status.retweet_count,
                'data': json.loads(status.json)}
            table.save(record)
        elif item is not None:
            record = {'_id': item[2], 'isActive': False}
            table.save(record)

    def _wait_between_requests(self, idx, length, download_pause_sec):
        # stay in Twitter API rate limits
        time_left_str = self.__get_time_left_str(idx, length, download_pause_sec)
        print '    pausing %f sec to obey Twitter API rate limits (%s left)' % \
              (download_pause_sec, time_left_str)
        time.sleep(download_pause_sec)

    def download_tweets_from_file(
            self,
            file_name=None,
            table_name='data',
            analyzer=None,
            tag=None,
            limit=0):

        table = self.db[table_name]

        if file_name is None:
            file_name = core.convert_rel_to_absolute(self.cfg.corpus_file)

        fetch_list = self.__read_tweet_list(file_name)
        length = len(fetch_list)

        if tag is None:
            tag = file_name

        if analyzer is None:
            analyzer = WordSentimentAnalyzer()

        for idx in range(0, length):
            if 0 < limit < idx:
                return
            item = fetch_list[idx]
            if table.find({'_id': item[2]}).count() == 0:
                try:
                    print '--> downloading tweet #%s (%d of %d)' % \
                          (item[2], idx + 1, length)
                    status = self.tweeter_api.get_status(id=item[2])
                except error.TweepError, e:
                    print 'ERROR - %s (Tweet: %s)' % \
                          (e.message[0], item[2])
                    self._save_tweet(table, None, False, None, tag, item)
                else:
                    self._save_tweet(table, status, True, analyzer, tag, item)
                finally:
                    self._wait_between_requests(idx, length, self._get_wait_time())
            else:
                print 'Tweet ', item[2], ' already downloaded!'
        return

    def download_tweets_using_query(
            self,
            query=None,
            limit=100,
            table_name='data',
            analyzer=None,
            tag=None,
            undersample=False):

        table = self.db[table_name]

        if analyzer is None:
            analyzer = WordSentimentAnalyzer()

        if limit is None:
            limit = 100

        if tag is None:
            if query is None:
                tag = "Timeline"
            else:
                tag = query

        if query is None or query == "":
            for tweet in tweepy.Cursor(self.tweeter_api.user_timeline, count=limit).items(limit):
                try:
                    self._save_tweet(table, tweet, True, analyzer, tag, undersample=undersample)
                except error.TweepError, e:
                    print "Error downloading tweet from timeline" + e.message[0]
        else:
            for tweet in tweepy.Cursor(self.tweeter_api.search, count=limit, q=query, lang='en').items(limit):
                try:
                    self._save_tweet(table, tweet, True, analyzer, tag, undersample=undersample)
                except tweepy.error.TweepError, e:
                    print "Error downloading tweet from timeline" + e.message[0]

        return