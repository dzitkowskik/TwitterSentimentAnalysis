# ERRORS
* 'tuple' object has no attribute 'items' in Introduction-to-Data-Mining-DTU/TwitterSentimentAnalysis/ai.py 
    in fill_with_predicted_data line:614 out.append(self.classifier.classify(rec)) 
                * using web site, try to run max entropy classification and save it with some name
* 'tuple' object has no attribute 'copy' in Introduction-to-Data-Mining-DTU/TwitterSentimentAnalysis/ai.py 
    in fill_with_predicted_data line:484 out.append(self.classifier.classify(rec))
                * using web site, try to run naive bayes classification and save it with some name    
* Cannot resolve keyword 'hour' into field. Choices are: ai, ai_id, day_of_week, id, sentiment_actual_avg, 
    sentiment_predicted_avg in Introduction-to-Data-Mining-DTU/site/tsa/tweets/statistics.py in get_day_of_week 
    line:197    
* test_max_ent_classifier is not passing: self.assertEqual(result, expected_error) -> AssertionError: 0.76 != 0.0
* test_naive_bayes_classifier is not passing: self.assertEqual(result, expected_error) -> 
    AssertionError: 0.56 != 0.0
        
        
# BUGS 
* For different AI there are different error units, for some there is accuracy, for some error rate in
    percents, and some have error rate as float from 0 to 1 or -1 do 1. For example: Error = -0.0252838173686
        * FOR CLASSIFICATION: IT SHOULD BE ALWAYS AN ERROR RATE AS % FROM 0 TO 100 WHERE 0% IS EQUAL CLASSIFICATION
        * FOR REGRESSION: IT SHOULD BE "Mean squared error regression loss" using for example
            sklearn.metrics.mean_squared_error function
            
* Predicted retweet count is sometimes lower than 0 and is a float, but it should not. It must be casted to integer
    and rounded to the nearest positive value. [-16.04641704] should be 0 and [ 21.64036913] should be 22.    
         
# TODO
* AI classifiers and regression algorithms should be changed to train until convergence with maximal epoch 
    count equal to 100 or better 50.
    
    
    