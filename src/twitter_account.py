import openai
import logging
from datetime import datetime
import pprint

class TwitterAccount:
    def __init__(self, user_id, username, tweet, retweet_count, mention_count,
                 follower_count, verified, bot_label, location, created_at, hashtags):
        logging.info("Initializing TwitterAccount")
        self.user_id = user_id
        self.username = username
        self.tweet = tweet
        self.retweet_count = retweet_count
        self.mention_count = mention_count
        self.follower_count = follower_count
        self.verified = verified
        self.bot_label = bot_label
        self.location = location
        self.created_at = created_at
        self.hashtags = hashtags.split(',') if hashtags else []
        self.avg_daily_retweets = self._calculate_avg_daily_retweets()
        logging.info(f"Account details: {self.get_account_details()}")

    def _calculate_avg_daily_retweets(self):
        try:
            creation_datetime = datetime.strptime(self.created_at, '%Y-%m-%d %H:%M:%S')
            days_active = (datetime.now() - creation_datetime).days
            avg = round(self.retweet_count / max(days_active, 1), 2)
            logging.debug(f"Average daily retweets calculated: {avg}")
            return avg
        except ValueError:
            logging.warning("Invalid date format. Using 0 for average daily retweets.")
            return 0

    def get_account_details(self):
        return {
            'user_id': self.user_id,
            'username': self.username,
            'tweet': self.tweet,
            'retweet_count': self.retweet_count,
            'mention_count': self.mention_count,
            'follower_count': self.follower_count,
            'verified': self.verified,
            'bot_label': self.bot_label,
            'location': self.location,
            'created_at': self.created_at,
            'hashtags': self.hashtags,
            'avg_daily_retweets': self.avg_daily_retweets
        }
    
    def __str__(self):
        return pprint.pformat(self.get_account_details())
