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
"""
        response = self._get_completion(prompt, "You are an impartial judge evaluating expert arguments.")
        return response.choices[0].message.content

    def _format_account_details(self, account_details):
        """Helper to format account details consistently."""
        return f"""Username: @{account_details['username']}
Account Creation: {account_details['created_at']}
Total Tweets: {account_details['tweet']}
Followers: {account_details['follower_count']}
Retweets: {account_details['retweet_count']}
Mentions: {account_details['mention_count']}
Verified: {'Yes' if account_details['verified'] else 'No'}
Location: {account_details['location']}
Hashtags: {', '.join(account_details['hashtags'])}
Average Daily Retweets: {account_details['avg_daily_retweets']}"""

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

