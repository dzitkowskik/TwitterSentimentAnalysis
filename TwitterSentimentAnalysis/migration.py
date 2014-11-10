# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import core
from downloaders import TweetDownloader


def main():
    print "Downloading tweets...\n"

    TweetDownloader().download_tweets_from_file()

    print '\nDownloading tweets done!\n'
    return


if __name__ == '__main__':
    core.initialize()
    main()
    core.terminate()
