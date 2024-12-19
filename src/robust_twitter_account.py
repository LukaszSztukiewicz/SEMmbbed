import logging
from datetime import datetime

class RobustTwitterAccount:
    def __init__(self, username, handle, description, location, webpage, 
                 joined, following, followers, tweets, is_bot):
        self.username = username
        self.handle = handle
        self.description = description
        self.location = location
        self.webpage = webpage
        self.joined = joined
        self.following = int(following)
        self.followers = int(followers)
        self.tweets = tweets if tweets else []
        self.bot_label = bool(int(is_bot))
        
    def get_account_details(self):
        return {
            'username': self.username,
            'handle': self.handle,
            'description': self.description,
            'location': self.location,
            'webpage': self.webpage,
            'joined': self.joined,
            'following': self.following,
            'followers': self.followers,
            'tweets': self.tweets,
            'is_bot': self.bot_label
        }
    
    def __str__(self):
        return f"{self.username} (@{self.handle})"