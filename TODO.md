
- NN training to predict retweet rate

- Analysis sentiment and retweet correlation

- Implement a dataset "factory" for regression (in datasets.py)

- Implement Naive Bayes and some other AI for regression of 
    retweet_count and classification of sentiment
    
- Refactoring: Change neuralNetworks.py to ai.py and 
    NeuralNetwork class to AI class (abstract classes)
    
- Write unit tests for datasets.py in datasets_test.py for all classes
    especially for get_dataset methods
    
- Move migration.py content to setup.py (necessary content should be downloaded
    during the application setup process)
     
- tweetSentiment.py is now only for custom testing purposes and should
    be deleted after implementing analysis functionality in the site
     
- write statistics functionality in statistics.py file which will produce
    some data necessary for making charts, for example some (<retweet_count> X <sentiment>)
    charts selecting some tag (tweet set) and some sentiment classification
    + Think about it how such a package can be implemented (some interface)