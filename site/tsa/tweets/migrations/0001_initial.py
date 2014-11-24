# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='ArtificialIntelligence',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(max_length=50)),
                ('tag', models.CharField(max_length=50)),
                ('path', models.CharField(max_length=250)),
                ('ai_type', models.CharField(max_length=100)),
                ('problem_type', models.IntegerField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='DayofweekRetweet',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('retweet_actual_avg', models.FloatField()),
                ('retweet_predicted_avg', models.FloatField()),
                ('hour', models.IntegerField()),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='DayofweekSentiment',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('sentiment_actual_avg', models.IntegerField()),
                ('sentiment_predicted_avg', models.IntegerField()),
                ('day_of_week', models.IntegerField()),
                ('ai', models.ForeignKey(to='tweets.ArtificialIntelligence')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='HourRetweet',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('retweet_actual_avg', models.FloatField()),
                ('retweet_predicted_avg', models.FloatField()),
                ('day_of_week', models.IntegerField()),
                ('ai', models.ForeignKey(to='tweets.ArtificialIntelligence')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='HourSentiment',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('sentiment_actual_avg', models.IntegerField()),
                ('sentiment_predicted_avg', models.IntegerField()),
                ('hour', models.IntegerField()),
                ('ai', models.ForeignKey(to='tweets.ArtificialIntelligence')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.CreateModel(
            name='Tweet',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('number', models.CharField(max_length=30)),
                ('text', models.TextField()),
                ('date', models.DateTimeField(null=True)),
                ('hour', models.IntegerField()),
                ('day_of_week', models.IntegerField()),
                ('favourites_count', models.IntegerField()),
                ('followers_count', models.IntegerField()),
                ('retweet_count_actual', models.IntegerField()),
                ('retweet_count_estimated', models.IntegerField(null=True)),
                ('sentiment_actual', models.CharField(max_length=30)),
                ('sentiment_estimated', models.CharField(max_length=30, null=True)),
                ('ai', models.ForeignKey(to='tweets.ArtificialIntelligence')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
