from django.contrib import admin
from models import Tweet, ArtificialIntelligence


class TweetModelAdmin(admin.ModelAdmin):
    pass


class ArtificialIntelligenceModelAdmin(admin.ModelAdmin):
    pass


admin.site.register(Tweet, TweetModelAdmin)
admin.site.register(ArtificialIntelligence, ArtificialIntelligenceModelAdmin)
