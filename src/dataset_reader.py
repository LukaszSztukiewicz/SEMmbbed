import csv
from .twitter_account import TwitterAccount

def read_dataset(filepath, limit=None):
    accounts = []
    with open(filepath, mode='r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            account = TwitterAccount(
                user_id=int(row['User ID']),
                username=row['Username'],
                tweet=row['Tweet'],
                retweet_count=int(row['Retweet Count']),
                mention_count=int(row['Mention Count']),
                follower_count=int(row['Follower Count']),
                verified=row['Verified'].strip().lower() == 'true',
                bot_label=int(row['Bot Label']),
                location=row['Location'],
                created_at=row['Created At'],
                hashtags=row['Hashtags']
            )
            accounts.append(account)
    return accounts[:limit] if limit else accounts