import tweepy
from config import Config

# Get config
f = file('configuration.cfg')
cfg = Config(f)


# == OAuth Authentication ==
#
# This mode of authentication is the new preferred way
# of authenticating with Twitter.

auth = tweepy.OAuthHandler(cfg.consumer_key, cfg.consumer_secret)
auth.secure = True
auth.set_access_token(cfg.access_token, cfg.access_token_secret)

api = tweepy.API(auth)

# If the authentication was successful, you should
# see the name of the account print out
print "My name is: " + api.me().name
print "List of my friends: "
for friend in api.friends():
    print friend.name + " " + str(friend.friends_count)

public_tweets = api.home_timeline()
for tweet in public_tweets:
    print tweet.text



# If the application settings are set for "Read and Write" then
# this line should tweet out the message to your account's 
# timeline. The "Read and Write" setting is on https://dev.twitter.com/apps
# api.update_status('Updating using OAuth authentication via Tweepy!')