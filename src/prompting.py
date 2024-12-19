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

def create_robust_analysis_prompt(account):
    """Create analysis prompt for OpenAI using RobustTwitterAccount data."""
    logging.info(f"Creating robust analysis prompt for {account}")
    
    tweet_text = "\n".join([f"Tweet {i+1}: {tweet}" for i, tweet in enumerate(account.tweets)])
    
    return f"""Analyze this Twitter account for bot-like behavior:
Username: @{account.username}
Handle: {account.handle}
Description: {account.description}
Location: {account.location}
Webpage: {account.webpage}
Joined: {account.joined}
Following: {account.following}
Followers: {account.followers}

Recent Tweets:
{tweet_text}

Based on these characteristics and tweet content, provide a detailed analysis of whether this account shows signs of being a bot or a genuine human user."""