from django.conf.urls import patterns, include, url
from django.contrib import admin
# noinspection PyUnresolvedReferences
from tweets.views import TweetSearchView

urlpatterns = patterns('',
                       url(r'^$', TweetSearchView.as_view()),
                       # url(r'^blog/', include('blog.urls')),

                       url(r'^admin/', include(admin.site.urls)),)
