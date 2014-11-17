from django.contrib import admin
from models import Tweet, Statistic


class TweetModelAdmin(admin.ModelAdmin):
    pass


class StatisticModelAdmin(admin.ModelAdmin):
    pass


admin.site.register(Tweet, TweetModelAdmin)
admin.site.register(Statistic, StatisticModelAdmin)
