import logging
import os
import time
from dotenv import load_dotenv
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from .dataset_reader import read_dataset
from .robust_dataset_reader import read_robust_dataset
from .openai_interface import OpenAIInterface, RobustOpenAIInterface
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm  # Add this import

# Configure logging to output to a file
logging.basicConfig(
    filename=f'logs/bot_detector_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    filemode='w',  # Overwrite the log file each time
    level=logging.DEBUG,  # Set to DEBUG to capture all messages
    format='%(asctime)s %(levelname)s:%(message)s'
)

def process_account(account, openai_interface):
    try:
        account_details = account.get_account_details()
        true_label = account.bot_label

        # Get initial arguments
        bot_args = openai_interface.get_bot_agent_arguments(account_details)
        human_args = openai_interface.get_human_agent_arguments(account_details)

        # Get critiques
        bot_critique = openai_interface.get_bot_critic_response(account_details, human_args)
        human_critique = openai_interface.get_human_critic_response(account_details, bot_args)

        # Get final judgment
        final_classification = openai_interface.get_final_classification(
            account_details, bot_args, human_args, bot_critique, human_critique)
        classification = openai_interface.get_classification_result_from_text(final_classification)

        logging.info(f"Account @{account.username} classified as {'Bot' if classification else 'Human'}")
        return true_label, classification
    except Exception as e:
        logging.error(f"Error analyzing account @{account.username}: {str(e)}")
        return true_label, 0  # Default to Human in case of error

def main():
    # Load environment variables from .env
    load_dotenv()
    api_key = os.getenv("API_KEY")
    model_name = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    limit_samples_dataset = os.getenv("LIMIT_SAMPLES_DATASET")
    temperature = float(os.getenv("TEMPERATURE", 0.5))
    use_robust = os.getenv("USE_ROBUST", "true").lower() == "true"

    # Initialize appropriate OpenAI interface
    if use_robust:
        openai_interface = RobustOpenAIInterface(api_key, model_name, temperature)
        dataset_path = os.getenv("ROBUST_DATASET_PATH", "data/robust_dataset.csv")
        accounts = read_robust_dataset(dataset_path)
    else:
        openai_interface = OpenAIInterface(api_key, model_name, temperature)
        dataset_path = os.getenv("DATASET_PATH", "dataset.csv")
        accounts = read_dataset(dataset_path, int(limit_samples_dataset) if limit_samples_dataset else None)

    logging.info(f"Twitter Bot Detector Started in {'robust' if use_robust else 'standard'} mode")
    print("\nTwitter Bot Detector Performance Evaluation")
    print("==========================================\n")
    print(f"Using {'robust' if use_robust else 'standard'} mode")
    print(f"Total accounts to analyze: {len(accounts)}\n")

    true_labels = []
    predicted_labels = []
    max_workers = int(os.getenv("MAX_WORKERS", os.cpu_count() or 1))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_account, account, openai_interface): account for account in accounts}
        for future in tqdm(as_completed(futures), total=len(futures)):
            true, pred = future.result()
            true_labels.append(true)
            predicted_labels.append(pred)
            time.sleep(0.1)  # Short delay to prevent rapid requests

    print("\nPerformance Evaluation")
    print("======================\n")
    # print labels and predictions
    print(f"True Labels     : {true_labels}")
    print(f"Predicted Labels: {predicted_labels}\n")

    # count correct predictions
    correct = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred)

    # Calculate performance metrics
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, zero_division=0)
    recall = recall_score(true_labels, predicted_labels, zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, zero_division=0)

    # Output performance metrics
    print("Performance Metrics:")
    print(f"Accuracy : {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall   : {recall:.2f}")
    print(f"F1 Score : {f1:.2f}")

    logging.info("Performance evaluation completed successfully")

if __name__ == "__main__":
    main()