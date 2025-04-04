import os
import sys
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# Add parent directory to path to import the ml_pipeline module
sys.path.append(str(Path(__file__).parent.parent.parent))

from ml_pipeline.data_loader import DataLoader
from ml_pipeline.data_processor import DataProcessor
from ml_pipeline.lgbm_model import LGBMModel
from ml_pipeline.feature_importance import FeatureImportance

def test_feature_importance():
    """Test the FeatureImportance class"""
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
        
        # Preprocess the full dataset
        print("Preprocessing data...")
        X_transformed, y = processor.fit_transform_full_dataset(data)
        
        # Try to load an existing model if available
        model_path = "test_outputs/target_classification_lgbm.pkl"
        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}...")
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        else:
            # Train a simple model for testing
            print("Training a model for testing...")
            model_config_test = model_config.copy()
            model_config_test["num_hp_searches"] = 2  # Reduce for faster testing
            
            lgbm_model = LGBMModel(
                model_config=model_config_test,
                output_dir="test_outputs"
            )
            
            # Train on full dataset with default parameters
            lgbm_model.train_with_full_dataset(X_transformed, y)
            model = lgbm_model.model
            
            # Save the model
            lgbm_model.save_model()
        
        # Get feature names
        feature_names = []
        for _, feature in feature_metadata.iterrows():
            feature_name = feature["feature_name"]
            feature_use = feature["use"]
            
            # Skip if not used in model or if it's an ID column or target
            if feature_use != 1 or feature_name in model_config.get("id_variables", []) or feature_name == model_config.get("target_variable", ""):
                continue
                
            feature_names.append(feature_name)
        
        # Test SHAP feature importance
        print("\nTesting SHAP feature importance...")
        try:
            # Use a small sample fraction for testing
            model_config_shap = model_config.copy()
            model_config_shap["importance_extraction_method"] = "shap"
            model_config_shap["sample_for_contribution"] = 0.2  # Use 20% of data for faster testing
            
            shap_importance = FeatureImportance(
                model_config=model_config_shap,
                feature_names=feature_names,
                output_dir="test_outputs"
            )
            
            # Calculate importance
            shap_results = shap_importance.calculate_importance(model, X_transformed)
            
            # Check results
            importance_df = shap_importance.get_importance_df()
            print(f"  SHAP importance DataFrame shape: {importance_df.shape}")
            print("  Top 5 features by SHAP importance:")
            print(importance_df.head(5))
            
            # Save results
            results_path = shap_importance.save_importance_results()
            print(f"  SHAP importance results saved to: {results_path}")
            
            # Save values
            values_path = shap_importance.save_importance_values()
            print(f"  SHAP importance values saved to: {values_path}")
            
            # Plot importance (comment this for CI environments)
            # shap_importance.plot_feature_importance(n_features=10, save_path="test_outputs/shap_importance.png")
            
        except Exception as e:
            print(f"Error during SHAP feature importance test: {str(e)}")
            raise
        
        # Test treeinterpreter feature importance
        print("\nTesting treeinterpreter feature importance...")
        try:
            # Use a small sample fraction for testing
            model_config_ti = model_config.copy()
            model_config_ti["importance_extraction_method"] = "treeinterpreter"
            model_config_ti["sample_for_contribution"] = 0.2  # Use 20% of data for faster testing
            
            ti_importance = FeatureImportance(
                model_config=model_config_ti,
                feature_names=feature_names,
                output_dir="test_outputs"
            )
            
            # Calculate importance
            ti_results = ti_importance.calculate_importance(model, X_transformed)
            
            # Check results
            importance_df = ti_importance.get_importance_df()
            print(f"  Treeinterpreter importance DataFrame shape: {importance_df.shape}")
            print("  Top 5 features by treeinterpreter importance:")
            print(importance_df.head(5))
            
            # Save results
            results_path = ti_importance.save_importance_results()
            print(f"  Treeinterpreter importance results saved to: {results_path}")
            
            # Save values
            values_path = ti_importance.save_importance_values()
            print(f"  Treeinterpreter importance values saved to: {values_path}")
            
            # Plot importance (comment this for CI environments)
            # ti_importance.plot_feature_importance(n_features=10, save_path="test_outputs/ti_importance.png")
            
        except Exception as e:
            print(f"Error during treeinterpreter feature importance test: {str(e)}")
            raise
        
        print("\nFeatureImportance test completed successfully")
        
    except Exception as e:
        print(f"Error during FeatureImportance test: {str(e)}")
        raise

if __name__ == "__main__":
    test_feature_importance() 