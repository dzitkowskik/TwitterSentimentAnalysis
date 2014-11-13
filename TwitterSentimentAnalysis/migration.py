# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

from config import Config
import core
from downloaders import TweetDownloader


def main():
    print "Downloading tweets...\n"

    cfg = Config(core.configuration_file_path)
    TweetDownloader().download_tweets_from_file(table_name=cfg.train_tweets_table)

    print '\nDownloading tweets done!\n'
    return


if __name__ == '__main__':
    core.initialize()
    main()
    core.terminate()