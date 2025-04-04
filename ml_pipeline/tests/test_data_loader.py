import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import the ml_pipeline module
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_pipeline.data_loader import DataLoader

def test_data_loader_initialization():
    """Test that the DataLoader initializes correctly"""
    data_config_path = "data_config_train.json"
    model_config_path = "model_config.json"
    
    try:
        loader = DataLoader(data_config_path, model_config_path)
        print("DataLoader initialized successfully")
        
        # Print configurations
        print("\nData Config:")
        for key, value in loader.get_data_config().items():
            print(f"  {key}: {value}")
        
        print("\nModel Config:")
        for key, value in loader.get_model_config().items():
            if key != "hyperparameter_space":  # Skip printing hyperparameter space for brevity
                print(f"  {key}: {value}")
        
        # Print feature metadata summary
        feature_metadata = loader.get_feature_metadata()
        print("\nFeature Metadata Summary:")
        print(f"  Total features: {len(feature_metadata)}")
        
        # Count features by type
        type_counts = feature_metadata['type'].value_counts()
        print("\nFeature Types:")
        for type_name, count in type_counts.items():
            print(f"  {type_name}: {count}")
        
        # Count features by use
        use_counts = feature_metadata['use'].value_counts()
        print("\nFeature Usage:")
        for use, count in use_counts.items():
            print(f"  {use}: {count}")
        
        # Print loaded data info
        data = loader.get_data()
        print("\nLoaded Data Info:")
        print(f"  Shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
        print(f"  Data types: \n{data.dtypes}")
        
        print("\nDataLoader test completed successfully")
        
    except Exception as e:
        print(f"Error during DataLoader test: {str(e)}")
        raise

if __name__ == "__main__":
    test_data_loader_initialization() 