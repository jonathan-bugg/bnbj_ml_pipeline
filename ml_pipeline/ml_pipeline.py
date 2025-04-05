import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union

from ml_pipeline.data_loader import DataLoader
from ml_pipeline.data_processor import DataProcessor
from ml_pipeline.lgbm_model import LGBMModel
from ml_pipeline.feature_importance import FeatureImportance


class ML_Pipeline:
    """
    Main class for the ML pipeline that integrates data loading, preprocessing,
    model training, and feature importance analysis.
    """
    
    def __init__(self, modeling_config: str, data_config: str):
        """
        Initialize the ML pipeline.
        
        Args:
            modeling_config: Path to the modeling configuration file
            data_config: Path to the data configuration file
        """
        # Store configuration paths
        self.modeling_config_path = modeling_config
        self.data_config_path = data_config
        
        # Create output directory
        self.output_dir = "trained_model_outputs_path"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data loader
        self.data_loader = DataLoader(data_config, modeling_config)
        
        # Get configurations
        self.data_config = self.data_loader.get_data_config()
        self.model_config = self.data_loader.get_model_config()
        self.feature_metadata = self.data_loader.get_feature_metadata()
        
        # Determine if this is a training or testing run
        self.is_training = self.data_config.get("training", True)
        
        # Initialize data processor
        self.data_processor = DataProcessor(
            feature_metadata=self.feature_metadata,
            model_config=self.model_config,
            output_dir=self.output_dir
        )
        
        # Model variables
        self.model = None
        self.model_performances = []
        
    def run(self) -> None:
        """
        Run the ML pipeline based on the configuration.
        
        If in training mode, train a model and save it.
        If in testing mode, load a model and make predictions.
        """
        # Load data
        data = self.data_loader.get_data()
        
        if self.is_training:
            self._run_training(data)
        else:
            self._run_testing(data)
    
    def _run_training(self, data: pd.DataFrame) -> None:
        """
        Run the training pipeline.
        
        Args:
            data: Input data for training
        """
        print("Starting training pipeline...")
        
        # Process data for training with fold creation
        processed_data = self.data_processor.process_data(data, is_training=True, create_folds=True)
        
        # Initialize model based on configuration
        model_type = self.model_config.get("models", ["lgbm"])[0]
        problem_type = self.model_config.get("problem_type", "classification")
        print(f"Training {model_type} models for {problem_type} problem...")
        
        # Train model for each fold
        if model_type == "lgbm":
            self._train_lgbm_model(processed_data)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Retrain on full dataset if specified
        if self.model_config.get("retrain_with_whole_dataset", True):
            self._retrain_full_dataset(data)
        
        print("Training pipeline completed.")
    
    def _train_lgbm_model(self, processed_data: Dict[str, Any]) -> None:
        """
        Train LGBM model using cross-validation.
        
        Args:
            processed_data: Processed data with folds
        """
        # Initialize LGBM model
        lgbm_model = LGBMModel(
            model_config=self.model_config,
            output_dir=self.output_dir
        )
        
        # First, perform hyperparameter optimization using all folds
        lgbm_model.train(processed_data, tune_hyperparameters=True)

        # Save hyperparameter study results if available
        try:
            study_path = lgbm_model.save_hyperparameter_study()
            print(f"Hyperparameter study results saved to {study_path}")
        except Exception as e:
            print(f"Note: No hyperparameter study results available or error saving: {str(e)}")
        
        # Store best hyperparameters and overall CV performance
        best_params = lgbm_model.get_best_params()
        cv_performance = lgbm_model.get_performance_metrics().get('cv_average', None)
        
        print(f"Cross-validation average performance: {cv_performance}")
        print("Best hyperparameters:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    
    def _retrain_full_dataset(self, data: pd.DataFrame) -> None:
        """
        Retrain model on the full dataset.
        
        Args:
            data: Full training dataset
        """
        print("Retraining on full dataset...")
        
        # Process full dataset without folds
        processed_data = self.data_processor.process_data(data, is_training=True, create_folds=False)
        
        # Initialize model
        model_type = self.model_config.get("models", ["lgbm"])[0]
        
        if model_type == "lgbm":
            lgbm_model = LGBMModel(
                model_config=self.model_config,
                output_dir=self.output_dir
            )
            
            # Train on full dataset
            lgbm_model.train(processed_data, tune_hyperparameters=False)
            
            # Store model and save it
            self.model = lgbm_model.model
            model_path = lgbm_model.save_model()
            print(f"Model saved to {model_path}")
            
            # Calculate feature importance
            self._calculate_feature_importance(processed_data)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print("Full dataset retraining completed.")
    
    def _calculate_feature_importance(self, processed_data: Dict[str, Any]) -> None:
        """
        Calculate feature importance for the trained model.
        
        Args:
            processed_data: Processed data with X_transformed
        """
        if self.model is None:
            print("No trained model available for feature importance calculation.")
            return
        
        print("Calculating feature importance...")
        
        # Get feature names
        feature_names = []
        for _, feature in self.feature_metadata.iterrows():
            feature_name = feature["feature_name"]
            feature_use = feature["use"]
            
            # Skip if not used in model or if it's an ID column or target
            if feature_use != 1 or feature_name in self.model_config.get("id_variables", []) or feature_name == self.model_config.get("target_variable", ""):
                continue
                
            feature_names.append(feature_name)
        
        # Get transformed data
        X_transformed = processed_data['X_transformed']
        
        # Get importance method
        importance_method = self.model_config.get("importance_extraction_method", "shap")
        
        try:
            # Create feature importance calculator
            feature_importance = FeatureImportance(
                model_config=self.model_config,
                feature_names=feature_names,
                output_dir=self.output_dir
            )
            
            # Calculate importance
            importance_results = feature_importance.calculate_importance(self.model, X_transformed)
            
            # Save results
            results_path = feature_importance.save_importance_results()
            values_path = feature_importance.save_importance_values()
            
            print(f"Feature importance calculated using {importance_method}.")
            print(f"Results saved to {results_path}")
            print(f"Values saved to {values_path}")
            
            # Display top important features
            importance_df = feature_importance.get_importance_df()
            print("\nTop 10 important features:")
            print(importance_df.head(10))
            
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
    
    def _run_testing(self, data: pd.DataFrame) -> None:
        """
        Run the testing pipeline.
        
        Args:
            data: Input data for testing
        """
        print("Starting testing pipeline...")
        
        # Load preprocessor
        preprocessor_path = self.data_config.get("data_processor_file_path", "")
        print(f"Loading preprocessor from {preprocessor_path}")
        self.data_processor.load_preprocessor(preprocessor_path)
        
        # Process test data
        processed_data = self.data_processor.process_data(data, is_training=False)
        
        # Load model
        model_path = self.data_config.get("model_file_path", "")
        self._load_model(model_path)
        
        # Make predictions
        X_transformed = processed_data['X_transformed']
        id_data = processed_data['id_data']
        predictions = self.model.predict(X_transformed)
        
        # Add IDs to predictions
        predictions_df = self.data_processor.add_ids_to_predictions(predictions, id_data)
        
        # Save predictions
        output_path = self.data_config.get("predictions_output_path", "")
        predictions_df.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")
        
        print("Testing pipeline completed.")
    
    def _load_model(self, file_path: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            file_path: Path to the saved model file
        """
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {file_path}")
    
    def get_model_performances(self) -> List[Dict[str, Any]]:
        """
        Get the performance metrics for each fold.
        
        Returns:
            List of dictionaries containing performance metrics for each fold
        """
        return self.model_performances 