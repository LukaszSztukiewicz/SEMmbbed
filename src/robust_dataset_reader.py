import csv
import logging
import os
from pathlib import Path
from .robust_twitter_account import RobustTwitterAccount

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def read_robust_dataset(filepath):
    accounts = []
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return accounts
        
    try:
        with open(filepath, mode='r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(
                csvfile, 
                dialect='excel',
                quoting=csv.QUOTE_MINIMAL
            )
            # Count total rows properly handling quoted fields
            with open(filepath, 'r', encoding='utf-8') as countfile:
                total_rows = sum(1 for _ in csv.DictReader(countfile, dialect='excel'))
            logger.info(f"Processing {total_rows} accounts...")
            
            for i, row in enumerate(reader, 1):
                try:
                    tweets = [
                        row['tweet1'], row['tweet2'], row['tweet3'],
                        row['tweet4'], row['tweet5']
                    ]
                    tweets = [t for t in tweets if t]  # Remove empty tweets
                    
                    # Add debug logging for problematic rows
                    if not all(key in row for key in ['username', 'handle', 'description']):
                        logger.debug(f"Row {i} missing fields: {row}")
                        continue
                        
                    account = RobustTwitterAccount(
                        username=row['username'],
                        handle=row['handle'],
                        description=row['description'],
                        location=row['location'],
                        webpage=row['webpage'],
                        joined=row['joined'],
                        following=row['following'],
                        followers=row['followers'],
                        tweets=tweets,
                        is_bot=row['is_bot']
                    )
                    accounts.append(account)
                    
                    if i % 100 == 0:
                        logger.info(f"Processed {i}/{total_rows} accounts...")
                        
                except KeyError as e:
                    logger.error(f"Missing required field in row {i}: {e}")
                    logger.debug(f"Problematic row content: {row}")
                except ValueError as e:
                    logger.error(f"Invalid data format in row {i}: {e}")
                    logger.debug(f"Problematic row content: {row}")
                    
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        
    logger.info(f"Successfully loaded {len(accounts)} accounts")
    return accounts

def main():
    dataset_path = "data/robust_dataset.csv"
    
    logger.info(f"Reading dataset from: {dataset_path}")
    accounts = read_robust_dataset(str(dataset_path))
    
    # Print summary statistics
    bot_count = sum(1 for acc in accounts if acc.is_bot)
    logger.info(f"Total accounts: {len(accounts)}")
    logger.info(f"Bot accounts: {bot_count}")
    logger.info(f"Human accounts: {len(accounts) - bot_count}")

    # print all the accounts
    for i, acc in enumerate(accounts, 1):
        logger.info(f"Account {i}: {acc}")
        logger.info(f"Details: {acc.get_account_details()}")
        

if __name__ == "__main__":
    main()