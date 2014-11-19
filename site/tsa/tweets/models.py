from django.db import models


class ArtificialIntelligence(models.Model):
    name = models.CharField(max_length=50)
    path = models.CharField(max_length=250)
    ai_type = models.CharField(max_length=100)

    def __unicode__(self):
        return self.name


class Statistic(models.Model):
    name = models.CharField(max_length=50)
    problem_type = models.IntegerField()
    ai = models.ForeignKey(ArtificialIntelligence)
    data = models.CharField(max_length=50)

    def __unicode__(self):
        return self.name


class Tweet(models.Model):
    number = models.CharField(max_length=30)
    text = models.TextField()
    favourites_count = models.IntegerField()
    followers_count = models.IntegerField()

    retweetCount_actual = models.IntegerField()
    retweetCount_estimated = models.IntegerField()
    sentiment_actual = models.CharField(max_length=30)
    sentiment_estimated = models.CharField(max_length=30)

    statistic = models.ForeignKey(Statistic)

    def __unicode__(self):
        return self.number