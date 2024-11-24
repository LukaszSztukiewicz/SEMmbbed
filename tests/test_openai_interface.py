import pytest
from dotenv import load_dotenv
from src.openai_interface import OpenAIInterface
import os

@pytest.fixture
def openai_interface():
    load_dotenv()
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    return OpenAIInterface(api_key=api_key, model_name=model_name)

def test_init_openai_interface(openai_interface):
    assert openai_interface.model_name == "gpt-4o-mini"
    assert openai_interface.client.api_key is not None

def test_get_classification_result_from_text(openai_interface):
    # Test case where classification is 'Yes'
    text_yes = "Classification: Yes\nReason: The account exhibits bot-like behavior."
    classification = openai_interface.get_classification_result_from_text(text_yes)
    assert classification == 1

    # Test case where classification is 'No'
    text_no = "Classification: No\nReason: The account appears to be genuine."
    classification = openai_interface.get_classification_result_from_text(text_no)
    assert classification == 0

    # Test with different casing
    text_mixed_case = "Classification: yEs\nReason: Unusual activity detected."
    classification = openai_interface.get_classification_result_from_text(text_mixed_case)
    assert classification == 1

    # Test with leading/trailing spaces
    text_spaces = "Classification:   No   \nReason: No indicators of bot activity."
    classification = openai_interface.get_classification_result_from_text(text_spaces)
    assert classification == 0

    # Test with real generated text
    text_real_generated = """- **Classification:** No  
- **Reason:** Although the account shows low engagement metrics and nonsensical tweet content, these factors alone do not definitively classify it as a bot. The account's long-term presence since August 2020, engagement attempt consistency (although selective), and the unique username and specified location indicate human-like behavior. The variability in tweet content and the absence of automated retweets further suggest a genuine user who may prioritize quality interactions over sheer numbers, which is more characteristic of human users than bots. Therefore, despite certain suspicious characteristics, the account exhibits enough human-like traits to classify it as not a bot.
"""
    classification = openai_interface.get_classification_result_from_text(text_real_generated)
    assert classification == 0

    # Test with invalid format (should raise an IndexError)
    text_invalid = "Some irrelevant text"
    with pytest.raises(IndexError):
        classification = openai_interface.get_classification_result_from_text(text_invalid)

def test_get_arguments_for_bot(openai_interface):
    account_details = {
        'username': 'testuser',
        'created_at': '2021-01-01',
        'tweet': 100,
        'follower_count': 50,
        'retweet_count': 20,
        'mention_count': 5,
        'verified': False,
        'location': 'Testville',
        'hashtags': ['test', 'bot'],
        'avg_daily_retweets': 2
    }
    result = openai_interface.get_arguments_for_bot(account_details)
    assert isinstance(result, str)
    assert len(result) > 0

def test_get_arguments_against_bot(openai_interface):
    account_details = {
        'username': 'testuser',
        'created_at': '2021-01-01',
        'tweet': 100,
        'follower_count': 50,
        'retweet_count': 20,
        'mention_count': 5,
        'verified': True,
        'location': 'Testville',
        'hashtags': ['test', 'human'],
        'avg_daily_retweets': 3
    }
    result = openai_interface.get_arguments_against_bot(account_details)
    assert isinstance(result, str)
    assert len(result) > 0

def test_get_final_classification(openai_interface):
    account_details = {
        'username': 'testuser',
        'created_at': '2021-01-01',
        'tweet': 100,
        'follower_count': 50,
        'retweet_count': 20,
        'mention_count': 5,
        'verified': False,
        'location': 'Testville',
        'hashtags': ['test', 'bot'],
        'avg_daily_retweets': 2
    }
    bot_args = openai_interface.get_arguments_for_bot(account_details)
    human_args = openai_interface.get_arguments_against_bot(account_details)
    result = openai_interface.get_final_classification(account_details, bot_args, human_args)
    assert isinstance(result, str)
    assert len(result) > 0