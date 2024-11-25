import logging

def create_analysis_prompt(account_details):
    """Create a detailed prompt for OpenAI analysis including all account characteristics."""
    logging.info("Creating analysis prompt for OpenAI")
    return f"""Analyze this Twitter account for bot-like behavior:
Username: @{account_details['username']}
Account Creation: {account_details['creation_date']}
Total Tweets: {account_details['tweet_count']}
Followers: {account_details['followers']}
Following: {account_details['following']}
Profile Picture: {'Yes' if account_details['has_profile_pic'] else 'No'}
Bio: {'Yes' if account_details['has_bio'] else 'No'}
Average Daily Tweets: {account_details['avg_daily_tweets']}

Based on these characteristics, provide a detailed analysis of whether this account shows signs of being a bot or a genuine human user."""