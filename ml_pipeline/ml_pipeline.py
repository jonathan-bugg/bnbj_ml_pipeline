import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import lightgbm as lgbm

from ml_pipeline.data_loader import DataLoader
from ml_pipeline.data_processor import DataProcessor


class ML_Pipeline:
    """
    Main class for the ML pipeline that integrates data loading, preprocessing,
    model training, and prediction.
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
        self.predictions = None
        self.model_performances = []
        
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
        4. Saving the model and preprocessor
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
        
        # Train a model for each fold
        for fold_idx, fold_data in enumerate(preprocessed_data['folds']):
            print(f"Training fold {fold_idx + 1}/{len(preprocessed_data['folds'])}...")
            
            X_train = fold_data['X_train_transformed']
            y_train = fold_data['y_train']
            X_val = fold_data['X_val_transformed']
            y_val = fold_data['y_val']
            
            # Train model
            if model_type == "lgbm":
                model = self._train_lgbm_model(X_train, y_train, X_val, y_val)
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Evaluate model
            performance = self._evaluate_model(model, X_val, y_val)
            self.model_performances.append({
                'fold_idx': fold_idx,
                'performance': performance
            })
            
            print(f"Fold {fold_idx + 1} performance: {performance}")
    
    def _train_lgbm_model(self, X_train, y_train, X_val, y_val):
        """
        Train a LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Trained LightGBM model
        """
        problem_type = self.model_config.get("problem_type", "classification")
        
        # Set up LGBM parameters
        params = {
            'objective': 'binary' if problem_type == 'classification' else 'regression',
            'metric': 'auc' if problem_type == 'classification' else 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1
        }
        
        # Create dataset
        train_data = lgbm.Dataset(X_train, label=y_train)
        val_data = lgbm.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        model = lgbm.train(
            params,
            train_data,
            valid_sets=[train_data, val_data],
            num_boost_round=1000,
            early_stopping_rounds=50,
            verbose_eval=100
        )
        
        return model
    
    def _evaluate_model(self, model, X_val, y_val):
        """
        Evaluate a model on validation data.
        
        Args:
            model: Trained model
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Performance metric value
        """
        problem_type = self.model_config.get("problem_type", "classification")
        evaluation_metric = self.model_config.get("evaluation_metric", "rocauc")
        
        # Make predictions
        if problem_type == "classification":
            y_pred = model.predict(X_val)
            
            # Calculate performance metric
            if evaluation_metric == "rocauc":
                from sklearn.metrics import roc_auc_score
                return roc_auc_score(y_val, y_pred)
            elif evaluation_metric == "accuracy":
                from sklearn.metrics import accuracy_score
                return accuracy_score(y_val, y_pred > 0.5)
            else:
                raise ValueError(f"Unsupported evaluation metric: {evaluation_metric}")
        else:
            y_pred = model.predict(X_val)
            
            # Calculate performance metric
            if evaluation_metric == "rmse":
                from sklearn.metrics import mean_squared_error
                return mean_squared_error(y_val, y_pred, squared=False)
            elif evaluation_metric == "mae":
                from sklearn.metrics import mean_absolute_error
                return mean_absolute_error(y_val, y_pred)
            else:
                raise ValueError(f"Unsupported evaluation metric: {evaluation_metric}")
    
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
            problem_type = self.model_config.get("problem_type", "classification")
            
            # Set up LGBM parameters
            params = {
                'objective': 'binary' if problem_type == 'classification' else 'regression',
                'metric': 'auc' if problem_type == 'classification' else 'rmse',
                'boosting_type': 'gbdt',
                'learning_rate': 0.05,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }
            
            # Create dataset
            train_data = lgbm.Dataset(X_transformed, label=y)
            
            # Train model
            self.model = lgbm.train(
                params,
                train_data,
                num_boost_round=1000
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Save the model
        self._save_model()
        
        print("Full dataset retraining completed.")
    
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