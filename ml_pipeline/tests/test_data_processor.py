import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add parent directory to path to import the ml_pipeline module
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_pipeline.data_loader import DataLoader
from ml_pipeline.data_processor import DataProcessor

def test_data_processor():
    """Test the DataProcessor class"""
    # First, load the data using DataLoader
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
        
        # Test feature categorization
        print("\nFeature Categorization:")
        print(f"  Numerical features: {processor.numerical_features}")
        print(f"  Categorical features: {processor.categorical_features}")
        print(f"  Binary features: {processor.binary_features}")
        print(f"  One-hot features: {processor.one_hot_features}")
        print(f"  Leave as NA features: {processor.leave_as_na_features}")
        
        # Test fold creation
        print("\nCreating folds...")
        fold_indices = processor.create_folds(data)
        print(f"  Number of folds: {len(fold_indices)}")
        
        # Test preprocessing for training
        print("\nPreprocessing training data...")
        preprocessed_data = processor.preprocess_training_data(data, fold_indices)
        
        print("\nPreprocessed Data Info:")
        print(f"  Original data shape: {preprocessed_data['original_data'].shape}")
        print(f"  Features shape: {preprocessed_data['X'].shape}")
        print(f"  Target shape: {preprocessed_data['y'].shape}")
        print(f"  Number of folds: {len(preprocessed_data['folds'])}")
        
        # Print info for the first fold
        first_fold = preprocessed_data['folds'][0]
        print("\nFirst Fold Info:")
        print(f"  Fold index: {first_fold['fold_idx']}")
        print(f"  Training indices shape: {first_fold['train_idx'].shape}")
        print(f"  Validation indices shape: {first_fold['val_idx'].shape}")
        print(f"  X_train shape: {first_fold['X_train'].shape}")
        print(f"  X_train_transformed shape: {first_fold['X_train_transformed'].shape}")
        print(f"  X_val shape: {first_fold['X_val'].shape}")
        print(f"  X_val_transformed shape: {first_fold['X_val_transformed'].shape}")
        
        # Test full dataset preprocessing
        print("\nPreprocessing full dataset...")
        X_transformed, y = processor.fit_transform_full_dataset(data)
        print(f"  X_transformed shape: {X_transformed.shape}")
        print(f"  y shape: {y.shape}")
        
        # Save preprocessor
        print("\nSaving preprocessor...")
        preprocessor_path = processor.save_preprocessor()
        print(f"  Preprocessor saved to: {preprocessor_path}")
        
        # Test the preprocessor on some test data
        # For this test, we'll just reuse the training data as if it were test data
        print("\nTesting preprocessor on test data...")
        data_without_target = data.copy()
        
        # Use the saved preprocessor
        processor.load_preprocessor(preprocessor_path)
        
        # Process test data
        X_test, id_data = processor.preprocess_test_data(data)
        print(f"  X_test shape: {X_test.shape}")
        print(f"  ID columns: {list(id_data.keys())}")
        
        # Create a dummy predictions array for testing
        print("\nTesting adding IDs to predictions...")
        dummy_predictions = np.random.random(X_test.shape[0])
        predictions_df = processor.add_ids_to_predictions(dummy_predictions, id_data)
        print(f"  Predictions DataFrame shape: {predictions_df.shape}")
        print(f"  Predictions DataFrame columns: {list(predictions_df.columns)}")
        
        print("\nDataProcessor test completed successfully")
        
    except Exception as e:
        print(f"Error during DataProcessor test: {str(e)}")
        raise

if __name__ == "__main__":
    test_data_processor() 