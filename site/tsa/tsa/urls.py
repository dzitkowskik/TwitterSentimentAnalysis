from django.conf.urls import patterns, include, url
from django.contrib import admin
# noinspection PyUnresolvedReferences
from tweets.views import *


# noinspection PyUnresolvedReferences
urlpatterns = patterns('',
                       url(r'^$', TweetSearchView.as_view()),
                       url(r'^analysis/$', AnalysisView.as_view()),
                       url(r'^contact/$', contact, name='contact'),
                       url(r'^admin/', include(admin.site.urls)), )
