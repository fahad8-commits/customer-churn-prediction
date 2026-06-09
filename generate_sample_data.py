import pandas as pd
import numpy as np
import os
from src.utils.logger import logger
from src.utils.config import config

def generate_sample_data(num_rows=1000):
    """Generates a sample dataset for demonstration purposes."""
    logger.info(f"Generating {num_rows} rows of sample data...")
    
    np.random.seed(42)
    
    data = {
        'timestamp': pd.date_range(start='2024-01-01', periods=num_rows, freq='h'),
        'feature_1': np.random.randn(num_rows),
        'feature_2': np.random.rand(num_rows) * 100,
        'category': np.random.choice(['A', 'B', 'C'], size=num_rows),
        'target': np.random.randint(0, 2, size=num_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Ensure raw data directory exists
    raw_data_dir = os.path.join(config.DATA_DIR, "raw")
    os.makedirs(raw_data_dir, exist_ok=True)
    
    output_path = os.path.join(raw_data_dir, "sample_data.csv")
    df.to_csv(output_path, index=False)
    
    logger.info(f"Sample data successfully saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    try:
        generate_sample_data()
    except Exception as e:
        logger.error(f"Failed to generate sample data: {e}")
