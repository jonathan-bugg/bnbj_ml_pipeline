import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from ml_pipeline.data_loader import DataLoader
from ml_pipeline.data_processor import DataProcessor
from ml_pipeline.lgbm_model import LGBMModel

def debug_data_processing():
    """Debug script to identify data conversion issues."""
    data_config_path = "data_config_train.json"
    model_config_path = "model_config.json"
    
    print("Initializing DataLoader...")
    loader = DataLoader(data_config_path, model_config_path)
    
    # Get configurations and data
    model_config = loader.get_model_config()
    feature_metadata = loader.get_feature_metadata()
    data = loader.get_data()
    
    print("\nData overview:")
    print(f"  Shape: {data.shape}")
    print(f"  Columns: {list(data.columns)}")
    
    # Print column types
    print("\nColumn data types:")
    for col in data.columns:
        print(f"  {col}: {data[col].dtype}")
    
    # Check for problematic values
    print("\nChecking for problematic values...")
    for col in data.columns:
        if data[col].dtype == 'object':  # String columns
            # Check if these should be numeric
            try:
                pd.to_numeric(data[col])
                print(f"  {col}: Can be converted to numeric")
            except Exception as e:
                # If can't be converted, look for problematic values
                if 'e' in ''.join(data[col].astype(str).tolist()):
                    print(f"  {col}: Contains 'e' character")
                    # Print a sample of values
                    sample_vals = data[col].sample(min(5, len(data[col]))).tolist()
                    print(f"    Sample values: {sample_vals}")
    
    print("\nInitializing DataProcessor...")
    processor = DataProcessor(
        feature_metadata=feature_metadata,
        model_config=model_config,
        output_dir="debug_outputs"
    )
    
    # Create folds for cross-validation
    print("\nCreating folds...")
    fold_indices = processor.create_folds(data)
    
    # Preprocess training data
    print("\nPreprocessing training data...")
    try:
        preprocessed_data = processor.preprocess_training_data(data, fold_indices)
        print("  Preprocessing successful")
        
        # Debug the first fold data
        fold_data = preprocessed_data['folds'][0]
        X_train = fold_data['X_train_transformed']
        y_train = fold_data['y_train']
        X_val = fold_data['X_val_transformed']
        y_val = fold_data['y_val']
        
        print(f"\nProcessed data shapes:")
        print(f"  X_train: {X_train.shape}, dtype: {X_train.dtype}")
        print(f"  y_train: {y_train.shape}, dtype: {y_train.dtype}")
        print(f"  X_val: {X_val.shape}, dtype: {X_val.dtype}")
        print(f"  y_val: {y_val.shape}, dtype: {y_val.dtype}")
        
        # Check for non-numeric values in X_train and X_val
        print("\nChecking for non-numeric values in processed data...")
        
        def check_numeric_array(arr, name):
            try:
                # Try converting to float
                float_arr = arr.astype(float)
                print(f"  {name}: All values can be converted to float")
            except Exception as e:
                print(f"  {name}: Error converting to float - {str(e)}")
                # Find problematic entries
                for i in range(arr.shape[0]):
                    for j in range(arr.shape[1]):
                        try:
                            float(arr[i, j])
                        except:
                            print(f"    Non-numeric value at {i},{j}: '{arr[i, j]}'")
                            # Only show a few examples
                            if i * arr.shape[1] + j > 10:
                                print("    (Showing only first few examples)")
                                return
        
        check_numeric_array(X_train, "X_train")
        check_numeric_array(X_val, "X_val")
        
    except Exception as e:
        print(f"  Error during preprocessing: {str(e)}")
        raise
    
if __name__ == "__main__":
    debug_data_processing() 