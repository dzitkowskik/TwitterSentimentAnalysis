from django.db import models


class Tweet(models.Model):
    number = models.CharField(max_length=30)
    isActive = models.BooleanField(default=True)
    text = models.TextField()
    retweetCount = models.IntegerField()
    jsonData = models.FileField()

    def __unicode__(self):
        return str(id)