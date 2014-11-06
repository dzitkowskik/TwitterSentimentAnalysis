from django.db import models


class Tweet(models.Model):
    id = models.CharField(max_length=30)
    isActive = models.BooleanField()
    text = models.TextField()
    retweetCount = models.IntegerField()
    jsonData = models.FileField()

    def __unicode__(self):
        return str(id)