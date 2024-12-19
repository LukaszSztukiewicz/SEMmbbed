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
        text = text + '\n'
        substr = "**Classification:**"

        # Find the classification substring
        start = text.find(substr)
        if start == -1:
            raise IndexError("Classification not found in OpenAI response")
        
        # Extract the classification value
        start += len(substr)
        end = text.find("\n", start)
        final_answer = text[start:end].strip()
        classification = None

        if final_answer.lower() == "yes":
            classification = 1
        elif final_answer.lower() == "no":
            classification = 0
        else:
            raise IndexError("Invalid classification format in OpenAI response")
        return classification

    def get_bot_agent_arguments(self, account_details):
        """Initial arguments for bot classification."""
        prompt = f"""\
As a Bot Detection Expert, analyze this Twitter account data and provide arguments suggesting it's a bot.

**Account Details:**
{self._format_account_details(account_details)}

Focus on suspicious patterns and red flags. Be thorough but concise.
"""
        response = self._get_completion(prompt, "You are an expert focused on detecting Twitter bots.")
        return response.choices[0].message.content

    def get_human_agent_arguments(self, account_details):
        """Initial arguments against bot classification."""
        prompt = f"""\
As a Human Behavior Expert, analyze this Twitter account data and provide arguments suggesting it's a genuine human user.

**Account Details:**
{self._format_account_details(account_details)}

Focus on authentic behavior patterns and human indicators. Be thorough but concise.
"""
        response = self._get_completion(prompt, "You are an expert in human social media behavior.")
        return response.choices[0].message.content

    def get_bot_critic_response(self, account_details, human_arguments):
        """Bot expert critiques the human agent's arguments."""
        prompt = f"""\
As a Bot Detection Expert, critique these arguments claiming the account is human:

**Account Details:**
{self._format_account_details(account_details)}

**Human Expert's Arguments:**
{human_arguments}

Point out flaws in their reasoning and provide counter-evidence.
"""
        response = self._get_completion(prompt, "You are a critical bot detection expert.")
        return response.choices[0].message.content

    def get_human_critic_response(self, account_details, bot_arguments):
        """Human expert critiques the bot agent's arguments."""
        prompt = f"""\
As a Human Behavior Expert, critique these arguments claiming the account is a bot:

**Account Details:**
{self._format_account_details(account_details)}

**Bot Expert's Arguments:**
{bot_arguments}

Point out flaws in their reasoning and provide counter-evidence.
"""
        response = self._get_completion(prompt, "You are a critical human behavior expert.")
        return response.choices[0].message.content

    def get_final_classification(self, account_details, bot_args, human_args, bot_critique, human_critique):
        """Judge makes final assessment based on the debate."""
        prompt = f"""\
As an impartial judge, review this Twitter account classification debate:

**Account Details:**
{self._format_account_details(account_details)}

**Initial Bot Arguments:**
{bot_args}

**Initial Human Arguments:**
{human_args}

**Bot Expert's Critique of Human Arguments:**
{bot_critique}

**Human Expert's Critique of Bot Arguments:**
{human_critique}

Carefully weigh all arguments and counter-arguments. Consider which side made stronger points and addressed the other's criticisms more effectively.

Provide your ruling with:
- **Analysis:** Evaluate the strength of each position and their critiques
- **Classification:** Yes/No (Is this account a bot?)

Keep the classification in this format for easy parsing:
Example 1: **Classification:** Yes
Example 2: **Classification:** No
Example 3: **Classification:** Yes
Example 4: **Classification:** No
"""
        response = self._get_completion(prompt, "You are an impartial judge evaluating expert arguments.")
        return response.choices[0].message.content

    def _format_account_details(self, account_details):
        """Helper to format account details consistently."""
        # Map from old to new field names
        field_mapping = {
            'created_at': 'joined',
            'follower_count': 'followers',
            'tweet': 'tweets',
            'retweet_count': None,
            'mention_count': None,
            'verified': None,
            'hashtags': None,
            'avg_daily_retweets': None
        }
        
        # Handle both dictionary and object formats
        if not isinstance(account_details, dict):
            account_details = account_details.get_account_details()
        
        # Build formatted string using available fields
        formatted = f"""Username: @{account_details['username']}
Location: {account_details['location']}"""

        # Add optional fields if they exist
        if 'joined' in account_details or 'created_at' in account_details:
            joined = account_details.get('joined', account_details.get('created_at', 'Unknown'))
            formatted += f"\nJoined: {joined}"
            
        if 'followers' in account_details or 'follower_count' in account_details:
            followers = account_details.get('followers', account_details.get('follower_count', 0))
            formatted += f"\nFollowers: {followers}"
            
        if 'following' in account_details:
            formatted += f"\nFollowing: {account_details['following']}"
            
        if 'tweets' in account_details:
            tweets = account_details['tweets']
            if isinstance(tweets, list):
                formatted += "\n\nRecent Tweets:"
                for i, tweet in enumerate(tweets):
                    formatted += f"\nTweet {i+1}: {tweet}"
                    
        return formatted

    def _get_completion(self, prompt, system_message):
        """Helper to get OpenAI completion with consistent parameters."""
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=self.temperature
        )

class RobustOpenAIInterface(OpenAIInterface):
    def _format_account_details(self, account):
        """Format account details for prompts, handling both dict and object formats."""
        if isinstance(account, dict):
            return super()._format_account_details(account)
            
        # Handle RobustTwitterAccount objects
        tweets_formatted = "\n".join([f"Tweet {i+1}: {tweet}" for i, tweet in enumerate(account.tweets)])
        
        return f"""Username: @{account.username}
Handle: {account.handle}
Description: {account.description}
Location: {account.location}
Webpage: {account.webpage}
Joined: {account.joined}
Following: {account.following}
Followers: {account.followers}

Recent Tweets:
{tweets_formatted}"""

    def get_bot_agent_arguments(self, account):
        """Generate bot detection arguments for RobustTwitterAccount."""
        prompt = f"""\
As a Bot Detection Expert, analyze this Twitter account:

{self._format_account_details(account)}

Provide arguments suggesting this is a bot account. Focus on:
1. Account metadata (followers, following, creation date)
2. Tweet content and patterns
3. Profile completeness
4. Behavioral indicators"""
        
        response = self._get_completion(prompt, "You are an expert in detecting Twitter bots.")
        return response.choices[0].message.content

    def get_human_agent_arguments(self, account):
        """Generate human behavior arguments for RobustTwitterAccount."""
        prompt = f"""\
As a Human Behavior Expert, analyze this Twitter account:

{self._format_account_details(account)}

Provide arguments suggesting this is a human account. Focus on:
1. Natural language patterns in tweets
2. Personal details and authenticity
3. Engagement patterns
4. Account history indicators"""
        
        response = self._get_completion(prompt, "You are an expert in human social media behavior.")
        return response.choices[0].message.content

    def get_final_classification(self, account, bot_args, human_args, bot_critique, human_critique):
        """Make final classification for RobustTwitterAccount."""
        prompt = f"""\
As an impartial judge, evaluate this Twitter account:

{self._format_account_details(account)}

Bot Expert Arguments:
{bot_args}

Human Expert Arguments:
{human_args}

Bot Expert's Critique:
{bot_critique}

Human Expert's Critique:
{human_critique}

Provide:
1. Analysis of all arguments
2. Assessment of tweet content authenticity
3. Final evaluation of account characteristics
4. **Classification:** Yes/No (Is this account a bot?)

Keep the classification in this format for easy parsing:
Example 1: **Classification:** Yes
Example 2: **Classification:** No
Example 3: **Classification:** Yes
Example 4: **Classification:** No

"""
        
        response = self._get_completion(prompt, "You are an impartial judge evaluating expert arguments.")
        return response.choices[0].message.content

