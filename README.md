# Social Media Bot Detection via Multi-Agent Debate

## About
Detect Twitter bots using LLM-powered multi-agent debate system. Agents analyze account behavior through structured dialogue to determine automation likelihood.

## Features
- Multi-agent debate architecture
- Parallel processing pipeline
- Configurable OpenAI model support
- Comprehensive logging
- Performance metrics

## Requirements
- Python 3.12+
- Conda
- OpenAI API key

## Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/social-bot-detector
cd social-bot-detector

# Setup environment
chmod +x setup_conda.sh
./setup_conda.sh

# Configure API key
cp .env.example .env
# Edit .env with your OpenAI key

# Run detector
conda activate twibot
python -m src.main
``` 

## Configuration
Edif .env file:
```bash
API_KEY=your_openai_api_key
MODEL_NAME=gpt-4
DATASET_PATH=data/bot_detection_data.csv
LIMIT_SAMPLES_DATASET=10
MAX_WORKERS=4
TEMPERATURE=0
```

## Project Structure
```
.
├── data/              # Datasets
├── logs/              # Log files
├── src/              # Source code
├── tests/            # Test suite
├── .env.example      # Config template
├── requirements.txt  # Dependencies
└── setup_conda.sh   # Setup script
```

## Performance

| Dataset  | Accuracy | Precision | Recall | F1 Score |
|:---------|:--------:|:---------:|:------:|:--------:|
| Standard | 0.49     | 0.50      | 0.64   | 0.56     |
| Robust   | 0.91     | 0.83      | 1.00   | 0.91     |

## Development
Run tests:
```bash
python -m pytest tests/
```

