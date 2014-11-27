from django.db import models


class ArtificialIntelligence(models.Model):
    name = models.CharField(max_length=50, unique=True)
    tag = models.CharField(max_length=50)
    path = models.CharField(max_length=250)
    ai_type = models.CharField(max_length=100)
    problem_type = models.IntegerField()

    def __unicode__(self):
        return self.name + " <=> " + self.ai_type


class Tweet(models.Model):
    number = models.CharField(max_length=30)
    text = models.TextField()
    date = models.DateTimeField(null=True)
    hour = models.IntegerField()
    day_of_week = models.IntegerField()
    favourites_count = models.IntegerField()
    followers_count = models.IntegerField()
    retweet_count_actual = models.IntegerField()
    retweet_count_estimated = models.IntegerField(null=True)
    sentiment_actual = models.CharField(max_length=30)
    sentiment_estimated = models.CharField(max_length=30, null=True)
    ai = models.ForeignKey(ArtificialIntelligence)


class HourSentiment(models.Model):
    sentiment_actual_avg = models.IntegerField()
    sentiment_predicted_avg = models.IntegerField(null=True)
    hour = models.IntegerField()
    ai = models.ForeignKey(ArtificialIntelligence)


class HourRetweet(models.Model):
    retweet_actual_avg = models.FloatField()
    retweet_predicted_avg = models.FloatField(null=True)
    hour = models.IntegerField()
    ai = models.ForeignKey(ArtificialIntelligence)


class DayofweekSentiment(models.Model):
    sentiment_actual_avg = models.IntegerField()
    sentiment_predicted_avg = models.IntegerField(null=True)
    day_of_week = models.IntegerField()
    ai = models.ForeignKey(ArtificialIntelligence)


class DayofweekRetweet(models.Model):
    retweet_actual_avg = models.FloatField()
    retweet_predicted_avg = models.FloatField(null=True)
    day_of_week = models.IntegerField()
    ai = models.ForeignKey(ArtificialIntelligence)

    def __unicode__(self):
        return self.number