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
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 output_dir: str = 'trained_model_outputs_path'):
        """
        Initialize the LGBMModel.
        
        Args:
            model_config: Dictionary containing the model configuration
            output_dir: Directory to save outputs to
        """
        self.model_config = model_config
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract model configuration
        self.target_variable = model_config.get("target_variable", "target")
        self.problem_type = model_config.get("problem_type", "classification")
        self.evaluation_metric = model_config.get("evaluation_metric", "rocauc")
        self.better_performance = model_config.get("better_performance", "gt")
        self.num_hp_searches = model_config.get("num_hp_searches", 10)
        self.hyperparameter_space = model_config.get("hyperparameter_space", {}).get("lgbm", {})
        
        # Model variables
        self.model = None
        self.best_params = None
        self.study_results = []
        self.performance_metrics = {}
    
    def train(self, 
              processed_data: Dict[str, Any],
              tune_hyperparameters: bool) -> None:
        """
        Train the model using the provided data.
        
        Args:
            processed_data: Dictionary containing processed data
            fold_idx: Index of the fold to use (None for full dataset training or when using k-fold CV in hyperparameter optimization)
        """
        if tune_hyperparameters:
            # Hyperparameter tuning using all folds
            print("Training with cross-validation hyperparameter optimization...")
            # Tune hyperparameters using all folds
            self._tune_hyperparameters_with_cv(processed_data['folds'])       
        else:
            # Training with full dataset
            if 'X_transformed' not in processed_data or 'y' not in processed_data:
                raise ValueError("Processed data must contain 'X_transformed' and 'y' for full dataset training.")
            
            X = processed_data['X_transformed']
            y = processed_data['y']
                        
            # Train model using best hyperparameters from previous tuning or defaults
            self._train_model(X, y)
    
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
    
    def _get_model_params(self, trial=None) -> Dict[str, Any]:
        """
        Get model parameters based on trial or best parameters.
        
        Args:
            trial: Optuna trial object (for hyperparameter optimization)
            
        Returns:
            Dictionary of model parameters
        """
        params = {}
        
        if trial is not None:
            # Parse hyperparameter space and suggest values for optimization
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
        elif self.best_params is not None:
            # Use best parameters from previous tuning
            params = self.best_params.copy()
        else:
            # Use default parameters
            params = {
                'boosting_type': 'gbdt',
                'learning_rate': 0.1,
                'num_leaves': 31,
                'feature_fraction': 0.8,
                'reg_lambda': 0.0,
                'n_estimators': 100,
                'min_data_in_leaf': 20,
                'class_weight': None
            }
        
        # Ensure integer parameters are correctly typed
        int_params = ['num_leaves', 'n_estimators', 'min_data_in_leaf']
        for param in int_params:
            if param in params:
                params[param] = int(params[param])
        
        # Map n_estimators to num_iterations for LightGBM if needed
        if 'n_estimators' in params:
            num_iterations = int(params.pop('n_estimators'))
            if trial is None:  # Only set num_iterations for actual training, not for trials
                params['num_iterations'] = num_iterations
        
        # Set objective based on problem type
        if self.problem_type == "classification":
            params["objective"] = "binary"
            params["metric"] = "auc"
        else:
            params["objective"] = "regression"
            params["metric"] = "rmse"
        
        return params
    
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
        # Get parameters for this trial
        params = self._get_model_params(trial)
        
        # Get num_iterations from n_estimators if it was in the params
        num_iterations = 100
        if 'n_estimators' in self.hyperparameter_space:
            for param_name, param_expr in self.hyperparameter_space.items():
                if param_name == 'n_estimators':
                    try:
                        num_iterations = int(eval(param_expr.replace('trial.suggest_discrete_uniform', 
                                                                    'trial.suggest_float')))
                    except:
                        num_iterations = 100
        
        # Ensure data is numeric
        try:
            # # Make sure X_train and X_val are in the right format for LightGBM
            # X_train_float = np.asarray(X_train, dtype=np.float64)
            # X_val_float = np.asarray(X_val, dtype=np.float64)
            
            # # Check for NaN or infinity values
            # if np.isnan(X_train_float).any() or np.isinf(X_train_float).any():
            #     print("Warning: NaN or Inf values found in X_train. Replacing with zeros.")
            #     X_train_float = np.nan_to_num(X_train_float, nan=0.0, posinf=0.0, neginf=0.0)
                
            # if np.isnan(X_val_float).any() or np.isinf(X_val_float).any():
            #     print("Warning: NaN or Inf values found in X_val. Replacing with zeros.")
            #     X_val_float = np.nan_to_num(X_val_float, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create dataset
            train_data = lgbm.Dataset(X_train, label=y_train)
            val_data = lgbm.Dataset(X_val, label=y_val, reference=train_data)
        except Exception as e:
            print(f"Error preparing data for LightGBM: {str(e)}")
            # Return a default poor performance value rather than crashing
            # Use 0.5 for AUC and similar metrics (this is "random" performance)
            return 0.5
        
        try:
            # Train model with early stopping
            params["verbosity"] = -1
            callbacks = [lgbm.early_stopping(stopping_rounds=50, verbose=False)]
            
            model = lgbm.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=num_iterations,
                callbacks=callbacks
            )
            
            # Predict on validation set and evaluate
            y_pred = model.predict(X_val_float)
            performance = self._evaluate_model(y_val, y_pred)
            
            return performance
        except Exception as e:
            print(f"Error in LightGBM training: {str(e)}")
            # Return a default poor performance value rather than crashing
            return 0.5
    
    def _train_model(self, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray) -> None:
        """
        Train the model using the best hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        print("Training model with best hyperparameters...")
        
        # Get model parameters
        params = self._get_model_params()
        
        # Ensure data is numeric
        try:
            # # Make sure X_train is in the right format for LightGBM
            # X_train_float = np.asarray(X_train, dtype=np.float64)
            
            # # Check for NaN or infinity values
            # if np.isnan(X_train_float).any() or np.isinf(X_train_float).any():
            #     print("Warning: NaN or Inf values found in X_train. Replacing with zeros.")
            #     X_train_float = np.nan_to_num(X_train_float, nan=0.0, posinf=0.0, neginf=0.0)
            
            # # Handle validation data if provided
            # X_val_float = None
            # if X_val is not None:
            #     X_val_float = np.asarray(X_val, dtype=np.float64)
            #     if np.isnan(X_val_float).any() or np.isinf(X_val_float).any():
            #         print("Warning: NaN or Inf values found in X_val. Replacing with zeros.")
            #         X_val_float = np.nan_to_num(X_val_float, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Create dataset
            train_data = lgbm.Dataset(X_train, label=y_train)
            
            # Set up validation if provided
            valid_sets = [train_data]
            callbacks = []
            
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
            
            # Evaluate model
            y_pred_train = self.model.predict(X_train)
            train_performance = self._evaluate_model(y_train, y_pred_train)
            self.performance_metrics['train'] = train_performance
            
            # if X_val_float is not None and y_val is not None:
            #     y_pred_val = self.model.predict(X_val_float)
            #     val_performance = self._evaluate_model(y_val, y_pred_val)
            #     self.performance_metrics['val'] = val_performance
            
            # Print performance
            print(f"Training {self.evaluation_metric}: {self.performance_metrics['train']}")
            if 'val' in self.performance_metrics:
                print(f"Validation {self.evaluation_metric}: {self.performance_metrics['val']}")
                
        except Exception as e:
            print(f"Error in model training: {str(e)}")
            # Set default performance metrics
            self.performance_metrics['train'] = 0.5
            if X_val is not None:
                self.performance_metrics['val'] = 0.5
    
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
    
    def _objective_cv(self, 
                   trial: optuna.Trial, 
                   folds: List[Dict[str, Any]]) -> float:
        """
        Objective function for Optuna hyperparameter optimization using cross-validation.
        
        Args:
            trial: Optuna trial
            folds: List of fold data dictionaries
            
        Returns:
            Average performance metric across all folds (to be maximized or minimized)
        """
        # Get parameters for this trial
        params = self._get_model_params(trial)
        
        # Get num_iterations from n_estimators if it was in the params
        num_iterations = 100
        if 'n_estimators' in self.hyperparameter_space:
            for param_name, param_expr in self.hyperparameter_space.items():
                if param_name == 'n_estimators':
                    try:
                        num_iterations = int(eval(param_expr.replace('trial.suggest_discrete_uniform', 
                                                                   'trial.suggest_float')))
                    except:
                        num_iterations = 100
        
        # Track performance for each fold
        fold_performances = []
        
        # Train and evaluate on each fold
        for fold_idx, fold_data in enumerate(folds):
            X_train = fold_data['X_train_transformed']
            y_train = fold_data['y_train']
            X_val = fold_data['X_val_transformed']
            y_val = fold_data['y_val']
            
            # Try to prepare data and train model
            try:
                # # Make sure X_train and X_val are in the right format for LightGBM
                # X_train_float = np.asarray(X_train, dtype=np.float64)
                # X_val_float = np.asarray(X_val, dtype=np.float64)
                
                # # Check for NaN or infinity values
                # if np.isnan(X_train_float).any() or np.isinf(X_train_float).any():
                #     print(f"Warning: NaN or Inf values found in X_train for fold {fold_idx}. Replacing with zeros.")
                #     X_train_float = np.nan_to_num(X_train_float, nan=0.0, posinf=0.0, neginf=0.0)
                    
                # if np.isnan(X_val_float).any() or np.isinf(X_val_float).any():
                #     print(f"Warning: NaN or Inf values found in X_val for fold {fold_idx}. Replacing with zeros.")
                #     X_val_float = np.nan_to_num(X_val_float, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Create dataset
                train_data = lgbm.Dataset(X_train, label=y_train)
                val_data = lgbm.Dataset(X_val, label=y_val, reference=train_data)
                
                # Train model with early stopping
                params_copy = params.copy()
                params_copy["verbosity"] = -1  # Quiet mode
                
                callbacks = [lgbm.early_stopping(stopping_rounds=50, verbose=False)]
                
                model = lgbm.train(
                    params_copy,
                    train_data,
                    valid_sets=[val_data],
                    num_boost_round=num_iterations,
                    callbacks=callbacks
                )
                
                # Predict on validation set and evaluate
                y_pred = model.predict(X_val)
                performance = self._evaluate_model(y_val, y_pred)
                fold_performances.append(performance)
                
            except Exception as e:
                print(f"Error in fold {fold_idx}: {str(e)}")
                fold_performances.append(0.5)  # Default poor performance
        
        # Calculate average performance across all folds
        avg_performance = sum(fold_performances) / len(fold_performances)
        return avg_performance
    
    def _tune_hyperparameters_with_cv(self, folds: List[Dict[str, Any]]) -> None:
        """
        Tune hyperparameters using Optuna with cross-validation.
        
        Args:
            folds: List of fold data dictionaries
        """
        print("Starting hyperparameter tuning with Optuna using cross-validation...")
        
        # Define the optimization direction based on better_performance
        direction = "maximize" if self.better_performance == "gt" else "minimize"
        
        # Create Optuna study
        study = optuna.create_study(direction=direction)
        
        # Define objective function for cross-validation
        objective_func = lambda trial: self._objective_cv(trial, folds)
        
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
        
        # Store the cross-validation performance
        self.performance_metrics = {
            'cv_average': study.best_trial.value,
            'best_trial': study.best_trial.number
        } 