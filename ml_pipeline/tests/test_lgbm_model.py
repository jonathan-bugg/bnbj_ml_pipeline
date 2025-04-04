import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import the ml_pipeline module
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_pipeline.data_loader import DataLoader
from ml_pipeline.data_processor import DataProcessor
from ml_pipeline.lgbm_model import LGBMModel

def test_lgbm_model():
    """Test the LGBMModel class"""
    data_config_path = "data_config_train.json"
    model_config_path = "model_config.json"
    
    try:
        print("Initializing DataLoader...")
        loader = DataLoader(data_config_path, model_config_path)
        
        # Get configurations and data
        model_config = loader.get_model_config()
        feature_metadata = loader.get_feature_metadata()
        data = loader.get_data()
        
        print("Initializing DataProcessor...")
        processor = DataProcessor(
            feature_metadata=feature_metadata,
            model_config=model_config,
            output_dir="test_outputs"
        )
        
        # Create folds for cross-validation
        fold_indices = processor.create_folds(data)
        
        # Preprocess training data
        print("Preprocessing training data...")
        preprocessed_data = processor.preprocess_training_data(data, fold_indices)
        
        # Initialize LGBM model
        print("Initializing LGBMModel...")
        # Set a low number of hyperparameter searches for testing
        model_config_test = model_config.copy()
        model_config_test["num_hp_searches"] = 2  # Reduce for faster testing
        
        lgbm_model = LGBMModel(
            model_config=model_config_test,
            output_dir="test_outputs"
        )
        
        # Test training with CV data
        print("\nTesting train_with_cv_data...")
        try:
            # Use fold 0 for training
            lgbm_model.train_with_cv_data(preprocessed_data, fold_idx=0)
            
            # Check that model is trained
            print(f"  Model trained: {lgbm_model.model is not None}")
            
            # Get performance metrics
            performance_metrics = lgbm_model.get_performance_metrics()
            print(f"  Performance metrics: {performance_metrics}")
            
            # Get best hyperparameters
            best_params = lgbm_model.get_best_params()
            print(f"  Best hyperparameters: {best_params}")
            
            # Test prediction
            fold_data = preprocessed_data['folds'][0]
            X_val = fold_data['X_val_transformed']
            y_pred = lgbm_model.predict(X_val)
            print(f"  Predictions shape: {y_pred.shape}")
            
            # Save model and hyperparameter study
            model_path = lgbm_model.save_model()
            print(f"  Model saved to: {model_path}")
            
            study_path = lgbm_model.save_hyperparameter_study()
            print(f"  Hyperparameter study saved to: {study_path}")
            
        except Exception as e:
            print(f"Error during train_with_cv_data test: {str(e)}")
            raise
        
        # Test training with full dataset
        print("\nTesting train_with_full_dataset...")
        try:
            # Preprocess the full dataset
            X_transformed, y = processor.fit_transform_full_dataset(data)
            
            # Initialize a new model for full dataset training
            lgbm_model_full = LGBMModel(
                model_config=model_config_test,
                output_dir="test_outputs"
            )
            
            # Train on full dataset (with best params from previous run)
            lgbm_model_full.best_params = lgbm_model.best_params
            lgbm_model_full.train_with_full_dataset(X_transformed, y)
            
            # Check that model is trained
            print(f"  Model trained: {lgbm_model_full.model is not None}")
            
            # Get performance metrics
            performance_metrics = lgbm_model_full.get_performance_metrics()
            print(f"  Performance metrics: {performance_metrics}")
            
            # Test prediction
            y_pred = lgbm_model_full.predict(X_transformed)
            print(f"  Predictions shape: {y_pred.shape}")
            
            # Save model
            model_path = lgbm_model_full.save_model(file_path="test_outputs/full_dataset_model.pkl")
            print(f"  Model saved to: {model_path}")
            
        except Exception as e:
            print(f"Error during train_with_full_dataset test: {str(e)}")
            raise
        
        print("\nLGBMModel test completed successfully")
        
    except Exception as e:
        print(f"Error during LGBMModel test: {str(e)}")
        raise

if __name__ == "__main__":
    test_lgbm_model() 