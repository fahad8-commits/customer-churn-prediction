# Demo Project 1

A structured data science project boilerplate with built-in logging, configuration management, and sample data generation.

## 📁 Project Structure

```text
.
├── data/               # Project data (raw, processed, etc.)
│   └── raw/            # Raw input data
├── logs/               # Application logs
├── notebooks/          # Jupyter notebooks for EDA
├── src/                # Source code
│   └── utils/          # Utility modules (config, logger)
├── tests/              # Unit and integration tests
├── .env                # Local environment variables (ignored by git)
├── .env.example        # Template for environment variables
├── .gitignore          # Git ignore file
├── generate_sample_data.py # Script to generate test data
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## 🚀 Getting Started

### 1. Setup Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configuration
Copy the example environment file and fill in your credentials:
```bash
cp .env.example .env
```
Edit `.env` to include your AWS credentials and project settings.

### 4. Generate Sample Data
Run the following command to generate a sample dataset in `data/raw/`:
```bash
export PYTHONPATH=$PYTHONPATH:.
python generate_sample_data.py
```

### 5. Start Jupyter Notebook
To start the Jupyter Notebook server (using local directories for configuration to avoid permission issues):
```bash
source .venv/bin/activate
export JUPYTER_CONFIG_DIR=$(pwd)/.jupyter_config
export JUPYTER_RUNTIME_DIR=$(pwd)/.jupyter_runtime
export JUPYTER_DATA_DIR=$(pwd)/.jupyter_data
export IPYTHONDIR=$(pwd)/.ipython
jupyter notebook --port=8889
```

## 🛠 Utilities

### Logger
Standardized logging to both console and file:
```python
from src.utils.logger import logger
logger.info("Your message here")
```

### Config
Centralized environment variable management:
```python
from src.utils.config import config
print(config.PROJECT_NAME)
```

## 🧪 Testing
Run tests using `pytest` (install it first if needed):
```bash
pytest tests/
```
