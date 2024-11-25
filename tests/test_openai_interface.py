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

def test_get_classification_result_from_text(openai_interface):
    # Test with real generated text
    text_real_generated = """- **Reason:** The account presents characteristics both supporting and opposing the classification as a bot. On the one hand, the disproportionate follower count relative to engagement and the presence of generic tweet content raise concerns about automation. However, the account's steady growth in followers, diverse content over a substantial period, and lack of an excessive automated engagement pattern suggest genuine human operation. The engagement patterns, though low, do not align with typical bot behavior that often focuses on amplification and visibility through excessive retweets and mentions. These human-like engagement tendencies, along with the unique but credible location information, point towards the account being operated by a human, despite potential atypical engagement metrics. Moreover, the account lacks characteristics like excessive hashtags, which are often indicative of bots trying to boost visibility artificially. Considering these factors, the balance tips towards a human-operated account.
- **Classification:** No
"""
    classification = openai_interface.get_classification_result_from_text(text_real_generated)
    assert classification == 0

    text_real_generated = """- **Reason:** The account presents characteristics both supporting and opposing the classification as a bot. On the one hand, the disproportionate follower count relative to engagement and the presence of generic tweet content raise concerns about automation. However, the account's steady growth in followers, diverse content over a substantial period, and lack of an excessive automated engagement pattern suggest genuine human operation. The engagement patterns, though low, do not align with typical bot behavior that often focuses on amplification and visibility through excessive retweets and mentions. These human-like engagement tendencies, along with the unique but credible location information, point towards the account being operated by a human, despite potential atypical engagement metrics. Moreover, the account lacks characteristics like excessive hashtags, which are often indicative of bots trying to boost visibility artificially. Considering these factors, the balance tips towards a human-operated account.
- **Classification:** Yes
"""
    classification = openai_interface.get_classification_result_from_text(text_real_generated)
    assert classification == 1

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