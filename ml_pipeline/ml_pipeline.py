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
        self.modeling_config_path = modeling_config
        self.data_config_path = data_config
        
        # Initialize data loader
        self.data_loader = DataLoader(data_config, modeling_config)
        
        # Get configurations
        self.data_config = self.data_loader.get_data_config()
        self.model_config = self.data_loader.get_model_config()
        self.feature_metadata = self.data_loader.get_feature_metadata()
        
        # Determine if this is a training or testing run
        self.is_training = self.data_config.get("training", True)
        
        # Create output directory if it doesn't exist
        self.output_dir = "trained_model_outputs_path"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize data processor
        self.data_processor = DataProcessor(
            feature_metadata=self.feature_metadata,
            model_config=self.model_config,
            output_dir=self.output_dir
        )
        
        # Variables to store model and predictions
        self.model = None
        self.lgbm_model = None  # Store the LGBMModel instance
        self.predictions = None
        self.model_performances = []
        self.feature_importance = None  # Store feature importance results
        
    def run(self) -> None:
        """
        Run the ML pipeline based on the configuration.
        
        If in training mode, train a model and save it.
        If in testing mode, load a model and make predictions.
        """
        if self.is_training:
            self._run_training()
        else:
            self._run_testing()
    
    def _run_training(self) -> None:
        """
        Run the training pipeline.
        
        This includes:
        1. Loading and preprocessing the training data
        2. Training a model using k-fold cross-validation
        3. Retraining on the full dataset if specified
        4. Calculating feature importance if specified
        5. Saving the model and preprocessor
        """
        print("Starting training pipeline...")
        
        # Get the processed data with correct types
        data = self.data_loader.get_data()
        
        # Create fold indices for cross-validation
        fold_indices = self.data_processor.create_folds(data)
        
        # Preprocess training data
        preprocessed_data = self.data_processor.preprocess_training_data(data, fold_indices)
        
        # Train models for each fold
        self._train_models(preprocessed_data)
        
        # Retrain on the full dataset if specified
        if self.model_config.get("retrain_with_whole_dataset", True):
            self._retrain_on_full_dataset(data)
            
            # Calculate feature importance if a model was trained
            if self.model is not None and self.lgbm_model is not None:
                self._calculate_feature_importance(data)
        
        print("Training pipeline completed.")
    
    def _run_testing(self) -> None:
        """
        Run the testing pipeline.
        
        This includes:
        1. Loading the test data
        2. Loading the trained model and preprocessor
        3. Preprocessing the test data
        4. Making predictions
        5. Saving the predictions
        """
        print("Starting testing pipeline...")
        
        # Get the processed data with correct types
        data = self.data_loader.get_data()
        
        # Load the preprocessor
        preprocessor_path = self.data_config.get("data_processor_file_path", "")
        self.data_processor.load_preprocessor(preprocessor_path)
        
        # Preprocess test data
        X_test, id_data = self.data_processor.preprocess_test_data(data)
  
        # Load the model
        model_path = self.data_config.get("model_file_path", "")
        self._load_model(model_path)
        
        # Make predictions
        self.predictions = self._predict(X_test)
        
        # Add IDs to predictions
        predictions_df = self.data_processor.add_ids_to_predictions(self.predictions, id_data)
        
        # Save predictions
        output_path = self.data_config.get("predictions_output_path", "")
        self._save_predictions(predictions_df, output_path)
        
        print("Testing pipeline completed.")
    
    def _train_models(self, preprocessed_data: Dict[str, Any]) -> None:
        """
        Train models for each fold using the preprocessed data.
        
        Args:
            preprocessed_data: Dictionary containing the preprocessed data for each fold
        """
        model_type = self.model_config.get("models", ["lgbm"])[0]
        problem_type = self.model_config.get("problem_type", "classification")
        
        print(f"Training {model_type} models for {problem_type} problem...")
        
        # Initialize the model
        if model_type == "lgbm":
            self.lgbm_model = LGBMModel(
                model_config=self.model_config,
                output_dir=self.output_dir
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Train a model for each fold
        for fold_idx, _ in enumerate(preprocessed_data['folds']):
            print(f"Training fold {fold_idx + 1}/{len(preprocessed_data['folds'])}...")
            
            # Train model using the LGBMModel class
            if model_type == "lgbm":
                self.lgbm_model.train_with_cv_data(preprocessed_data, fold_idx)
                
                # Get performance metrics for this fold
                fold_performance = self.lgbm_model.get_performance_metrics()
                
                # Store performance
                self.model_performances.append({
                    'fold_idx': fold_idx,
                    'performance': fold_performance.get('val', None)
                })
                
                print(f"Fold {fold_idx + 1} performance: {fold_performance.get('val', None)}")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
    
    def _retrain_on_full_dataset(self, data: pd.DataFrame) -> None:
        """
        Retrain a model on the full dataset.
        
        Args:
            data: Full training dataset
        """
        print("Retraining on full dataset...")
        
        # Preprocess the full dataset
        X_transformed, y = self.data_processor.fit_transform_full_dataset(data)

        # Train model on full dataset
        model_type = self.model_config.get("models", ["lgbm"])[0]
        
        if model_type == "lgbm":
            # Ensure we have a model instance
            if self.lgbm_model is None:
                self.lgbm_model = LGBMModel(
                    model_config=self.model_config,
                    output_dir=self.output_dir
                )
            
            # Train on full dataset
            self.lgbm_model.train_with_full_dataset(X_transformed, y)
            
            # Store the model
            self.model = self.lgbm_model.model
            
            # Save the model
            model_path = self.lgbm_model.save_model()
            print(f"Model saved to {model_path}")
            
            # Save hyperparameter study results if available
            try:
                if self.lgbm_model.study_results:
                    study_path = self.lgbm_model.save_hyperparameter_study()
                    print(f"Hyperparameter study results saved to {study_path}")
            except Exception as e:
                print(f"Error saving hyperparameter study: {str(e)}")
                
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        print("Full dataset retraining completed.")
    
    def _calculate_feature_importance(self, data: pd.DataFrame) -> None:
        """
        Calculate feature importance for the trained model.
        
        Args:
            data: Full training dataset
        """
        # Only proceed if we have a trained model
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
        
        # Get preprocessed data
        X_transformed, _ = self.data_processor.fit_transform_full_dataset(data)
        
        # Initialize feature importance with the specified method
        importance_method = self.model_config.get("importance_extraction_method", "shap")
        
        try:
            # Create feature importance calculator
            self.feature_importance = FeatureImportance(
                model_config=self.model_config,
                feature_names=feature_names,
                output_dir=self.output_dir
            )
            
            # Calculate importance
            importance_results = self.feature_importance.calculate_importance(self.model, X_transformed)
            
            # Save results
            results_path = self.feature_importance.save_importance_results()
            values_path = self.feature_importance.save_importance_values()
            
            print(f"Feature importance calculated using {importance_method}.")
            print(f"Results saved to {results_path}")
            print(f"Values saved to {values_path}")
            
            # Display top important features
            importance_df = self.feature_importance.get_importance_df()
            print("\nTop 10 important features:")
            print(importance_df.head(10))
            
        except Exception as e:
            print(f"Error calculating feature importance: {str(e)}")
    
    def _save_model(self, file_path: str = None) -> str:
        """
        Save the trained model to a file.
        
        Args:
            file_path: Path to save the model to (default: None)
            
        Returns:
            Path to the saved model file
        """
        if file_path is None:
            # Create a default file path
            target_var = self.model_config.get("target_variable", "target")
            problem_type = self.model_config.get("problem_type", "classification")
            model_type = self.model_config.get("models", ["lgbm"])[0]
            file_path = os.path.join(self.output_dir, f"{target_var}_{problem_type}_{model_type}.pkl")
        
        if self.model is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {file_path}")
        else:
            raise ValueError("Model has not been trained yet.")
        
        return file_path
    
    def _load_model(self, file_path: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            file_path: Path to the saved model file
        """
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {file_path}")
    
    def _predict(self, X_test) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X_test: Test features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained or loaded yet.")
        
        return self.model.predict(X_test)
    
    def _save_predictions(self, predictions_df: pd.DataFrame, file_path: str) -> None:
        """
        Save predictions to a CSV file.
        
        Args:
            predictions_df: DataFrame containing predictions
            file_path: Path to save the predictions to
        """
        predictions_df.to_csv(file_path, index=False)
        print(f"Predictions saved to {file_path}")
    
    def get_model_performances(self) -> List[Dict[str, Any]]:
        """
        Get the performance metrics for each fold.
        
        Returns:
            List of dictionaries containing performance metrics for each fold
        """
        return self.model_performances
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get the feature importance results.
        
        Returns:
            DataFrame containing feature importance results
        """
        if self.feature_importance is not None:
            return self.feature_importance.get_importance_df()
        return None 