# -*- coding: utf-8 -*-
from __future__ import unicode_literals

from django.db import models, migrations


class Migration(migrations.Migration):

    dependencies = [
        ('tweets', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ArtificialIntelligence',
            fields=[
                ('id', models.AutoField(verbose_name='ID', serialize=False, auto_created=True, primary_key=True)),
                ('name', models.CharField(max_length=50)),
                ('path', models.CharField(max_length=250)),
                ('ai_type', models.CharField(max_length=100)),
            ],
            options={
            },
            bases=(models.Model,),
        ),
        migrations.AlterField(
            model_name='statistic',
            name='ai',
            field=models.ForeignKey(to='tweets.ArtificialIntelligence'),
            preserve_default=True,
        ),
    ]
