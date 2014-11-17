- Analysis sentiment and retweet correlation

- Move migration.py content to setup.py (necessary content should be downloaded
    during the application setup process)
     
- tweetSentiment.py is now only for custom testing purposes and should
    be deleted after implementing analysis functionality in the site
     
- write statistics functionality in statistics.py file which will produce
    some data necessary for making charts, for example some (<retweet_count> X <sentiment>)
    charts selecting some tag (tweet set) and some sentiment classification
    + Think about it how such a package can be implemented (some interface)
    
- Implement loading suitable AI from file (ai.py)

- Implement filling data for other AIs (ai.py)