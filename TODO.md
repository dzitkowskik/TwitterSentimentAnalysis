# ERRORS
* 'tuple' object has no attribute 'items' in ```Introduction-to-Data-Mining-DTU/TwitterSentimentAnalysis/ai.py 
    in fill_with_predicted_data line:614 out.append(self.classifier.classify(rec)) ```
    * using web site, try to run max entropy classification and save it with some name
* 'tuple' object has no attribute 'copy' in ```Introduction-to-Data-Mining-DTU/TwitterSentimentAnalysis/ai.py 
    in fill_with_predicted_data line:484 out.append(self.classifier.classify(rec))```
    * using web site, try to run naive bayes classification and save it with some name    
* Cannot resolve keyword 'hour' into field. Choices are: ai, ai_id, day_of_week, id, sentiment_actual_avg, 
    sentiment_predicted_avg in ```Introduction-to-Data-Mining-DTU/site/tsa/tweets/statistics.py in get_day_of_week 
    line:197```
    
# NOT PASSING UNIT TESTS
* test_max_ent_classifier is not passing: ```self.assertEqual(result, expected_error) -> AssertionError: 0.76 != 0.0```
* test_naive_bayes_classifier is not passing: ```self.assertEqual(result, expected_error) -> 
    AssertionError: 0.56 != 0.0```
        
        
# BUGS     
* Plots also have a bugs which are begging for fixing! Lost values etc. Sentiment actual is sometimes always 0
    while the data actually contains different values.
         
# TODO
* AI classifiers and regression algorithms should be changed to train until convergence with maximal epoch 
    count equal to 100 or better 50.
    
    
    
