import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import the ml_pipeline module
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_pipeline.ml_pipeline import ML_Pipeline

def test_ml_pipeline_training():
    """Test the ML_Pipeline training functionality"""
    # Use the training configuration
    data_config_path = "data_config_train.json"
    model_config_path = "model_config.json"
    
    try:
        print("Initializing ML_Pipeline for training...")
        pipeline = ML_Pipeline(model_config_path, data_config_path)
        
        # Check that the pipeline was initialized correctly
        print("\nPipeline Initialization:")
        print(f"  Is training: {pipeline.is_training}")
        
        # Run the pipeline
        print("\nRunning training pipeline...")
        pipeline.run()
        
        # Check the model performances
        performances = pipeline.get_model_performances()
        print("\nModel Performances:")
        for perf in performances:
            print(f"  Fold {perf['fold_idx']}: {perf['performance']}")
        
        print("\nTraining ML_Pipeline test completed successfully")
        
    except Exception as e:
        print(f"Error during ML_Pipeline training test: {str(e)}")
        raise

def test_ml_pipeline_testing():
    """Test the ML_Pipeline testing functionality"""
    # Use the testing configuration
    data_config_path = "data_config_test.json"
    model_config_path = "model_config.json"
    
    try:
        print("Initializing ML_Pipeline for testing...")
        
        # Just check that configuration loading works correctly
        # We skip running the full pipeline in test mode since we 
        # don't have proper test files set up
        pipeline = ML_Pipeline(model_config_path, data_config_path)
        
        # Check that the pipeline was initialized correctly
        print("\nPipeline Initialization:")
        print(f"  Is training: {pipeline.is_training}")
        
        # Skip running the pipeline since we don't have test data files
        print("\nSkipping pipeline run in test mode since test data files are not set up.")
        print("Testing ML_Pipeline test initialization completed successfully")
        
    except Exception as e:
        print(f"Error during ML_Pipeline testing initialization: {str(e)}")
        raise

if __name__ == "__main__":
    # First run the training test
    test_ml_pipeline_training()
    
    # Then run the testing test
    test_ml_pipeline_testing() 