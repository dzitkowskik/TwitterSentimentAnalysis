# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Statistic',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(max_length=50)),
                ('problem_type', models.IntegerField()),
                ('ai', models.CharField(max_length=100)),
                ('data', models.CharField(max_length=50)),
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
                ('favourites_count', models.IntegerField()),
                ('followers_count', models.IntegerField()),
                ('retweetCount_actual', models.IntegerField()),
                ('retweetCount_estimated', models.IntegerField()),
                ('sentiment_actual', models.CharField(max_length=30)),
                ('sentiment_estimated', models.CharField(max_length=30)),
                ('statistic', models.ForeignKey(to='tweets.Statistic')),
            ],
            options={
            },
            bases=(models.Model,),
        ),
    ]
