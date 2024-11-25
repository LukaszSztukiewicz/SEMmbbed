import logging
from openai import OpenAI
from dotenv import load_dotenv
import os
from .prompting import create_analysis_prompt

class OpenAIInterface:
    def __init__(self, api_key, model_name, temperature=0.5):
        logging.info("Initializing OpenAI interface")
        load_dotenv()
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        logging.debug(f"OpenAI model set to: {self.model_name}, temperature: {self.temperature}")

    def get_classification_result_from_text(self, text):
        final_answer = text.split("**Classification:**")[1].strip()
        if final_answer.lower() == "yes":
            classification = 1
        elif final_answer.lower() == "no":
            classification = 0
        else:
            raise IndexError("Invalid classification format in OpenAI response")
        return classification

    def get_arguments_for_bot(self, account_details):
        """Get arguments supporting bot classification using OpenAI."""
        logging.info("Generating arguments for bot classification")
        prompt = f"""\
Analyze the following Twitter account data and identify reasons that suggest the account might be a bot.

**Account Details:**
Username: @{account_details['username']}
Account Creation: {account_details['created_at']}
Total Tweets: {account_details['tweet']}
Followers: {account_details['follower_count']}
Retweets: {account_details['retweet_count']}
Mentions: {account_details['mention_count']}
Verified: {'Yes' if account_details['verified'] else 'No'}
Location: {account_details['location']}
Hashtags: {', '.join(account_details['hashtags'])}
Average Daily Retweets: {account_details['avg_daily_retweets']}

**Provide your analysis with the following structure:**
1. Suspicious Patterns
2. Red Flags

**Example:**
1. Suspicious Patterns:
   - High retweet frequency suggests automation.
   - Lack of location information.
2. Red Flags:
   - Sudden spikes in follower count.

**Now, provide the arguments supporting the classification of this account as a bot."""

        # Log the prompt
        logging.debug(f"Prompt for bot arguments:\n{prompt}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in detecting Twitter bot accounts."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )

        # Log the response
        logging.debug(f"Response for bot arguments:\n{response.choices[0].message.content}")

        logging.debug("Bot arguments retrieved from OpenAI")
        return response.choices[0].message.content

    def get_arguments_against_bot(self, account_details):
        logging.info("Generating arguments against bot classification")
        prompt = f"""\
Analyze the following Twitter account data and identify reasons that suggest the account is operated by a genuine human.

**Account Details:**
Username: @{account_details['username']}
Account Creation: {account_details['created_at']}
Total Tweets: {account_details['tweet']}
Followers: {account_details['follower_count']}
Retweets: {account_details['retweet_count']}
Mentions: {account_details['mention_count']}
Verified: {'Yes' if account_details['verified'] else 'No'}
Location: {account_details['location']}
Hashtags: {', '.join(account_details['hashtags'])}
Average Daily Retweets: {account_details['avg_daily_retweets']}

**Provide your analysis with the following structure:**
1. Authentic Behavior Patterns
2. Indicators of Human Operation

**Example:**
1. Authentic Behavior Patterns:
   - Diverse tweet content indicates human creativity.
   - Engagement with followers through replies.
2. Indicators of Human Operation:
   - Verified status provides authenticity.
   - Consistent retweet patterns over time.

**Now, provide the arguments against the classification of this account as a bot."""

        # Log the prompt
        logging.debug(f"Prompt for human arguments:\n{prompt}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in detecting genuine human behavior on Twitter."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )

        # Log the response
        logging.debug(f"Response for human arguments:\n{response.choices[0].message.content}")

        return response.choices[0].message.content

    def get_final_classification(self, account_details, bot_args, human_args):
        logging.info("Generating final classification")
        prompt = f"""\
Based on the following analysis, provide a final classification of the Twitter account as a bot or not. Consider the arguments for and against bot classification. Make the decision carefully and provide a clear explanation for your choice.
Think carefully about the arguments presented and the account details provided. Write down the reasons for your classification and treat them more like a thought process than a simple answer. Only then proceed to provide the final classification.

**Account Details:**
Username: @{account_details['username']}
Account Creation: {account_details['created_at']}
Total Tweets: {account_details['tweet']}
Followers: {account_details['follower_count']}
Retweets: {account_details['retweet_count']}
Mentions: {account_details['mention_count']}
Verified: {'Yes' if account_details['verified'] else 'No'}
Location: {account_details['location']}
Hashtags: {', '.join(account_details['hashtags'])}
Average Daily Retweets: {account_details['avg_daily_retweets']}

**Bot Arguments:**
{bot_args}

**Human Arguments:**
{human_args}

**Provide your final classification with the following structure:**
- **Reason:** Detailed explanation based on the above arguments.
- **Classification:** Yes/No

**Example:**
- **Reason:** The account exhibits high retweet frequency and lacks location information, which are strong indicators of bot activity.
- **Classification:** Yes

**Now, provide the final classification for this account."""

        # Log the prompt
        logging.debug(f"Prompt for final classification:\n{prompt}")

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are an expert in summarizing analysis to provide clear classifications."},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )

        # Log the response
        logging.debug(f"Response for final classification:\n{response.choices[0].message.content}")

        return response.choices[0].message.content

