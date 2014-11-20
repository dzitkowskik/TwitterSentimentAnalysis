from django.db import models


class ArtificialIntelligence(models.Model):
    name = models.CharField(max_length=50)
    path = models.CharField(max_length=250)
    ai_type = models.CharField(max_length=100)
    problem_type = models.IntegerField()

    def __unicode__(self):
        return self.name


class Tweet(models.Model):
    number = models.CharField(max_length=30)
    text = models.TextField()
    favourites_count = models.IntegerField()
    followers_count = models.IntegerField()
    retweet_count_actual = models.IntegerField()
    retweet_count_estimated = models.IntegerField(null=True)
    sentiment_actual = models.CharField(max_length=30)
    sentiment_estimated = models.CharField(max_length=30, null=True)
    ai = models.ForeignKey(ArtificialIntelligence)

    def __unicode__(self):
        return self.number