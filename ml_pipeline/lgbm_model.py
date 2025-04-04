import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import optuna
import lightgbm as lgbm
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score


class LGBMModel:
    """
    Class for training LGBM models with hyperparameter tuning.
    
    This class trains LGBM models using Optuna for hyperparameter optimization.
    It supports cross-validation and retraining on the full dataset.
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 output_dir: str = 'trained_model_outputs_path'):
        """
        Initialize the LGBMModel.
        
        Args:
            model_config: Dictionary containing the model configuration
            output_dir: Directory to save outputs to (default: 'trained_model_outputs_path')
        """
        self.model_config = model_config
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store model attributes
        self.target_variable = model_config.get("target_variable", "target")
        self.problem_type = model_config.get("problem_type", "classification")
        self.evaluation_metric = model_config.get("evaluation_metric", "rocauc")
        self.better_performance = model_config.get("better_performance", "gt")
        self.num_hp_searches = model_config.get("num_hp_searches", 10)
        self.hyperparameter_space = model_config.get("hyperparameter_space", {}).get("lgbm", {})
        self.retrain_with_whole_dataset = model_config.get("retrain_with_whole_dataset", True)
        
        # Variables to store model and performances
        self.model = None
        self.best_params = None
        self.study_results = []
        self.performance_metrics = {}
        
    def _evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Evaluate the model using the specified evaluation metric.
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Performance metric value
        """
        if self.problem_type == "classification":
            if self.evaluation_metric == "rocauc":
                return roc_auc_score(y_true, y_pred)
            elif self.evaluation_metric == "recall" or self.evaluation_metric == "rec":
                # For recall, precision, and accuracy, we need binary predictions
                y_pred_binary = (y_pred > 0.5).astype(int)
                return recall_score(y_true, y_pred_binary)
            elif self.evaluation_metric == "precision" or self.evaluation_metric == "prec":
                y_pred_binary = (y_pred > 0.5).astype(int)
                return precision_score(y_true, y_pred_binary)
            elif self.evaluation_metric == "accuracy" or self.evaluation_metric == "acc":
                y_pred_binary = (y_pred > 0.5).astype(int)
                return accuracy_score(y_true, y_pred_binary)
            else:
                raise ValueError(f"Unsupported evaluation metric: {self.evaluation_metric}")
        else:
            # For regression
            if self.evaluation_metric == "rmse":
                from sklearn.metrics import mean_squared_error
                return mean_squared_error(y_true, y_pred, squared=False)
            elif self.evaluation_metric == "mae":
                from sklearn.metrics import mean_absolute_error
                return mean_absolute_error(y_true, y_pred)
            else:
                raise ValueError(f"Unsupported evaluation metric: {self.evaluation_metric}")
    
    def _objective(self, 
                  trial: optuna.Trial, 
                  X_train: np.ndarray, 
                  y_train: np.ndarray,
                  X_val: np.ndarray, 
                  y_val: np.ndarray) -> float:
        """
        Objective function for Optuna hyperparameter optimization.
        
        Args:
            trial: Optuna trial
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Performance metric value (to be maximized or minimized)
        """
        # Set up parameters for this trial
        params = {}
        
        # Parse hyperparameter space and suggest values
        for param_name, param_expr in self.hyperparameter_space.items():
            try:
                # Use eval to execute the parameter suggestion expression
                params[param_name] = eval(param_expr)
            except Exception as e:
                print(f"Error suggesting parameter {param_name}: {str(e)}")
                # Use default value if there's an error
                if param_name == "boosting_type":
                    params[param_name] = "gbdt"
                elif param_name == "learning_rate":
                    params[param_name] = 0.1
                elif param_name == "num_leaves":
                    params[param_name] = 31
                elif param_name == "feature_fraction":
                    params[param_name] = 0.8
                elif param_name == "reg_lambda":
                    params[param_name] = 0.0
                elif param_name == "n_estimators":
                    params[param_name] = 100
                elif param_name == "class_weight":
                    params[param_name] = None
                elif param_name == "min_data_in_leaf":
                    params[param_name] = 20
        
        # Ensure integer parameters are correctly typed
        int_params = ['num_leaves', 'n_estimators', 'min_data_in_leaf']
        for param in int_params:
            if param in params:
                params[param] = int(params[param])
        
        # Map n_estimators to num_iterations for LightGBM
        num_iterations = 1000
        if 'n_estimators' in params:
            num_iterations = int(params.pop('n_estimators'))
        
        # Set objective based on problem type
        if self.problem_type == "classification":
            params["objective"] = "binary"
            params["metric"] = "auc"
        else:
            params["objective"] = "regression"
            params["metric"] = "rmse"
        
        # Create dataset
        train_data = lgbm.Dataset(X_train, label=y_train)
        val_data = lgbm.Dataset(X_val, label=y_val, reference=train_data)
        
        # Train model
        params["verbosity"] = -1
        
        # Use callbacks for early stopping
        callbacks = [lgbm.early_stopping(stopping_rounds=50, verbose=False)]
        
        model = lgbm.train(
            params,
            train_data,
            valid_sets=[val_data],
            num_boost_round=num_iterations,
            callbacks=callbacks
        )
        
        # Predict on validation set
        y_pred = model.predict(X_val)
        
        # Calculate performance metric
        performance = self._evaluate_model(y_val, y_pred)
        
        return performance
    
    def tune_hyperparameters(self, 
                           X_train: np.ndarray, 
                           y_train: np.ndarray,
                           X_val: np.ndarray, 
                           y_val: np.ndarray) -> Dict[str, Any]:
        """
        Tune hyperparameters using Optuna.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary containing the best hyperparameters
        """
        print("Starting hyperparameter tuning with Optuna...")
        
        # Define the optimization direction based on better_performance
        direction = "maximize" if self.better_performance == "gt" else "minimize"
        
        # Create Optuna study
        study = optuna.create_study(direction=direction)
        
        # Define objective function for this dataset
        objective_func = lambda trial: self._objective(trial, X_train, y_train, X_val, y_val)
        
        # Run optimization
        study.optimize(objective_func, n_trials=self.num_hp_searches)
        
        # Store study results
        self.study_results = []
        for i, trial in enumerate(study.trials):
            self.study_results.append({
                'trial_number': i,
                'params': trial.params,
                'value': trial.value,
                'state': trial.state,
            })
        
        # Print optimization results
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best {self.evaluation_metric}: {study.best_trial.value}")
        print("Best hyperparameters:")
        for key, value in study.best_trial.params.items():
            print(f"  {key}: {value}")
        
        # Store best parameters
        self.best_params = study.best_trial.params
        
        return self.best_params
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None, 
             y_val: Optional[np.ndarray] = None) -> None:
        """
        Train the model using the best hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        print("Training model with best hyperparameters...")
        
        # If best parameters are not available, use defaults
        if self.best_params is None:
            print("No best parameters available. Using default hyperparameters.")
            self.best_params = {
                'boosting_type': 'gbdt',
                'learning_rate': 0.1,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'reg_lambda': 0.0,
                'n_estimators': 100,
                'min_data_in_leaf': 20
            }
        
        # Set up parameters
        params = self.best_params.copy()
        
        # Ensure integer parameters are correctly typed
        int_params = ['num_leaves', 'n_estimators', 'min_data_in_leaf']
        for param in int_params:
            if param in params:
                params[param] = int(params[param])
        
        # Map n_estimators to num_iterations for LightGBM
        if 'n_estimators' in params:
            params['num_iterations'] = int(params.pop('n_estimators'))
        
        # Set objective based on problem type
        if self.problem_type == "classification":
            params["objective"] = "binary"
            params["metric"] = "auc"
        else:
            params["objective"] = "regression"
            params["metric"] = "rmse"
        
        # Create dataset
        train_data = lgbm.Dataset(X_train, label=y_train)
        
        # If validation data is provided, use it
        valid_sets = [train_data]
        callbacks = []
        
        if X_val is not None and y_val is not None:
            val_data = lgbm.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            
            # Use callbacks for early stopping and logging
            callbacks = [
                lgbm.early_stopping(stopping_rounds=50, verbose=True),
                lgbm.log_evaluation(period=100)
            ]
        
        # Train model
        self.model = lgbm.train(
            params,
            train_data,
            valid_sets=valid_sets,
            callbacks=callbacks
        )
        
        # Evaluate model on training data
        y_pred_train = self.model.predict(X_train)
        train_performance = self._evaluate_model(y_train, y_pred_train)
        self.performance_metrics['train'] = train_performance
        
        # If validation data is provided, evaluate on it
        if X_val is not None and y_val is not None:
            y_pred_val = self.model.predict(X_val)
            val_performance = self._evaluate_model(y_val, y_pred_val)
            self.performance_metrics['val'] = val_performance
        
        # Print performance
        print(f"Training {self.evaluation_metric}: {self.performance_metrics['train']}")
        if 'val' in self.performance_metrics:
            print(f"Validation {self.evaluation_metric}: {self.performance_metrics['val']}")
    
    def train_with_cv_data(self, 
                         preprocessed_data: Dict[str, Any],
                         fold_idx: int = 0) -> None:
        """
        Train the model using cross-validation data from a specific fold.
        
        Args:
            preprocessed_data: Dictionary containing the preprocessed data for each fold
            fold_idx: Index of the fold to use for training (default: 0)
        """
        if fold_idx >= len(preprocessed_data['folds']):
            raise ValueError(f"Fold index {fold_idx} is out of range. There are only {len(preprocessed_data['folds'])} folds.")
        
        # Get data for the specified fold
        fold_data = preprocessed_data['folds'][fold_idx]
        X_train = fold_data['X_train_transformed']
        y_train = fold_data['y_train']
        X_val = fold_data['X_val_transformed']
        y_val = fold_data['y_val']
        
        # Tune hyperparameters
        self.tune_hyperparameters(X_train, y_train, X_val, y_val)
        
        # Train model with best hyperparameters
        self.train(X_train, y_train, X_val, y_val)
    
    def train_with_full_dataset(self, 
                              X: pd.DataFrame, 
                              y: pd.Series) -> None:
        """
        Train the model on the full dataset.
        
        Args:
            X: Features
            y: Target
        """
        print("Training on full dataset...")
        
        # Train model with best hyperparameters
        self.train(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X: Features
            
        Returns:
            Array of predictions
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        
        return self.model.predict(X)
    
    def save_model(self, file_path: str = None) -> str:
        """
        Save the trained model to a file.
        
        Args:
            file_path: Path to save the model to (default: None)
            
        Returns:
            Path to the saved model file
        """
        if file_path is None:
            # Create a default file path
            target_var = self.target_variable
            problem_type = self.problem_type
            file_path = os.path.join(self.output_dir, f"{target_var}_{problem_type}_lgbm.pkl")
        
        if self.model is not None:
            with open(file_path, 'wb') as f:
                pickle.dump(self.model, f)
            print(f"Model saved to {file_path}")
        else:
            raise ValueError("Model has not been trained yet.")
        
        return file_path
    
    def save_hyperparameter_study(self, file_path: str = None) -> str:
        """
        Save the hyperparameter tuning study results to a CSV file.
        
        Args:
            file_path: Path to save the study results to (default: None)
            
        Returns:
            Path to the saved study results file
        """
        if file_path is None:
            # Create a default file path
            target_var = self.target_variable
            problem_type = self.problem_type
            file_path = os.path.join(self.output_dir, f"{target_var}_{problem_type}_lgbm_hyperparameter_study.csv")
        
        if self.study_results:
            # Convert study results to DataFrame
            study_df = pd.DataFrame(self.study_results)
            
            # Save to CSV
            study_df.to_csv(file_path, index=False)
            print(f"Hyperparameter study results saved to {file_path}")
        else:
            raise ValueError("No hyperparameter study results available.")
        
        return file_path
    
    def load_model(self, file_path: str) -> None:
        """
        Load a trained model from a file.
        
        Args:
            file_path: Path to the saved model file
        """
        with open(file_path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {file_path}")
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get the performance metrics of the trained model.
        
        Returns:
            Dictionary containing performance metrics
        """
        return self.performance_metrics
    
    def get_study_results(self) -> List[Dict[str, Any]]:
        """
        Get the hyperparameter tuning study results.
        
        Returns:
            List of dictionaries containing study results
        """
        return self.study_results
    
    def get_best_params(self) -> Dict[str, Any]:
        """
        Get the best hyperparameters from the tuning study.
        
        Returns:
            Dictionary containing the best hyperparameters
        """
        return self.best_params 