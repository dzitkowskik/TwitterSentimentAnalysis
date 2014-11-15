from django.db import models


class Tweet(models.Model):
    number = models.CharField(max_length=30)
    text = models.TextField()
    favourites_count = models.IntegerField()
    followers_count = models.IntegerField()
    retweetCount_actual = models.IntegerField()
    sentiment_actual = models.CharField(max_length=30)
    retweetCount_estimated = models.IntegerField()
    sentiment_estimated = models.CharField(max_length=30)

    def __unicode__(self):
        return self.number