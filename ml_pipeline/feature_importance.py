import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import shap
from treeinterpreter import treeinterpreter as ti
import matplotlib.pyplot as plt


class FeatureImportance:
    """
    Class for calculating feature importance using SHAP or treeinterpreter.
    
    This class calculates feature importance for LGBM models using either SHAP or
    treeinterpreter methods.
    """
    
    def __init__(self, 
                 model_config: Dict[str, Any],
                 feature_names: List[str] = None,
                 output_dir: str = 'trained_model_outputs_path'):
        """
        Initialize the FeatureImportance.
        
        Args:
            model_config: Dictionary containing the model configuration
            feature_names: List of feature names
            output_dir: Directory to save outputs to (default: 'trained_model_outputs_path')
        """
        self.model_config = model_config
        self.feature_names = feature_names
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract configuration
        self.target_variable = model_config.get("target_variable", "target")
        self.importance_method = model_config.get("importance_extraction_method", "shap")
        self.sample_fraction = model_config.get("sample_for_contribution", 1.0)
        
        # Storage for results
        self.importance_values = None
        self.importance_df = None
        self.feature_importance_rankings = None
    
    def calculate_importance(self, 
                          model: Any, 
                          X: np.ndarray,
                          method: str = None) -> Dict[str, Any]:
        """
        Calculate feature importance using the specified method.
        
        Args:
            model: Trained model
            X: Features to calculate importance for
            method: Method to use (overrides self.importance_method if provided)
            
        Returns:
            Dictionary containing feature importance results
        """
        importance_method = method if method else self.importance_method
        
        # Sample data if needed
        X_sample = self._prepare_sample(X)
        
        # Calculate feature importance using specified method
        try:
            if importance_method.lower() == "shap":
                return self._calculate_shap_importance(model, X_sample)
            elif importance_method.lower() == "treeinterpreter":
                return self._calculate_treeinterpreter_importance(model, X_sample)
            else:
                print(f"Unsupported importance method: {importance_method}. Falling back to SHAP.")
                return self._calculate_shap_importance(model, X_sample)
        except Exception as e:
            print(f"Error calculating feature importance with {importance_method}: {str(e)}")
            # If specified method fails, try the other method
            if importance_method.lower() == "shap":
                print("Trying treeinterpreter instead...")
                try:
                    return self._calculate_treeinterpreter_importance(model, X_sample)
                except Exception as e2:
                    print(f"Error with treeinterpreter fallback: {str(e2)}")
                    raise ValueError("Both SHAP and treeinterpreter methods failed.")
            else:
                print("Falling back to SHAP...")
                return self._calculate_shap_importance(model, X_sample)
    
    def _prepare_sample(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare a sample of data for feature importance calculation.
        
        Args:
            X: Input features
            
        Returns:
            Sampled features array
        """
        # If sample_fraction is less than 1.0, sample a subset of the data
        if self.sample_fraction < 1.0:
            if self.sample_fraction <= 0.0:
                raise ValueError("sample_fraction must be greater than 0.")
            
            n_samples = int(X.shape[0] * self.sample_fraction)
            if n_samples < 1:
                n_samples = 1
            
            # Randomly sample rows
            sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sample = X[sample_indices]
            
            print(f"Using {n_samples} samples ({self.sample_fraction * 100:.1f}% of data) for feature importance calculation")
        else:
            X_sample = X
            print(f"Using all {X.shape[0]} samples for feature importance calculation")
            
        return X_sample
    
    def _process_importance_values(self, feature_importance_values: np.ndarray, method: str) -> Dict[str, Any]:
        """
        Process the raw importance values into a DataFrame and rankings.
        
        Args:
            feature_importance_values: Raw importance values
            method: Method used to calculate importance
            
        Returns:
            Dictionary containing processed results
        """
        # Store the raw values
        self.importance_values = feature_importance_values
        
        # Calculate mean absolute values for each feature
        mean_abs_values = np.abs(feature_importance_values).mean(axis=0)
        
        # Create DataFrame with feature names and importance values
        if self.feature_names is not None:
            # Make sure feature_names length matches
            if len(self.feature_names) != mean_abs_values.shape[0]:
                print(f"Warning: feature_names length ({len(self.feature_names)}) doesn't match "
                      f"number of features ({mean_abs_values.shape[0]}). Using generic feature names.")
                feature_names = [f"feature_{i}" for i in range(mean_abs_values.shape[0])]
            else:
                feature_names = self.feature_names
                
            # Create DataFrame
            self.importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': mean_abs_values
            })
        else:
            # Create with generic names
            self.importance_df = pd.DataFrame({
                'feature': [f"feature_{i}" for i in range(mean_abs_values.shape[0])],
                'importance': mean_abs_values
            })
        
        # Sort by importance
        self.importance_df = self.importance_df.sort_values('importance', ascending=False)
        
        # Create feature rankings
        self.feature_importance_rankings = self.importance_df.copy()
        self.feature_importance_rankings['rank'] = range(1, len(self.feature_importance_rankings) + 1)
        
        # Return results dictionary
        results = {
            'method': method,
            'values': feature_importance_values,
            'mean_abs_values': mean_abs_values,
            'importance_df': self.importance_df,
            'rankings': self.feature_importance_rankings
        }
        
        return results
    
    def _calculate_shap_importance(self, model: Any, X: np.ndarray) -> Dict[str, Any]:
        """
        Calculate feature importance using SHAP.
        
        Args:
            model: Trained model
            X: Features to calculate importance for
            
        Returns:
            Dictionary containing SHAP feature importance results
        """
        print("Calculating SHAP feature importance...")
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X)
            
            # For classification with multiple classes, shap_values will be a list
            if isinstance(shap_values, list):
                # For binary classification, we usually want the positive class
                shap_values = shap_values[-1]
            
            # Process and return results
            return self._process_importance_values(shap_values, 'shap')
        except Exception as e:
            print(f"Error in SHAP calculation: {str(e)}")
            raise
    
    def _calculate_treeinterpreter_importance(self, model: Any, X: np.ndarray) -> Dict[str, Any]:
        """
        Calculate feature importance using treeinterpreter.
        
        Args:
            model: Trained model
            X: Features to calculate importance for
            
        Returns:
            Dictionary containing treeinterpreter feature importance results
        """
        print("Calculating treeinterpreter feature importance...")
        
        # For LightGBM models, we need to create a sklearn compatible model
        if hasattr(model, 'params') and 'objective' in model.params:
            print("Converting LightGBM Booster to sklearn-compatible model for treeinterpreter...")
            try:
                model = self._convert_lightgbm_to_sklearn(model, X)
            except Exception as e:
                print(f"Error converting LightGBM model: {str(e)}")
                print("Falling back to SHAP...")
                return self._calculate_shap_importance(model, X)
        
        try:
            # Check if the model is a compatible type for treeinterpreter
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
            if not isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
                print("Model is not a DecisionTreeClassifier or DecisionTreeRegressor, which is required by treeinterpreter.")
                print("Falling back to SHAP...")
                return self._calculate_shap_importance(model, X)
                
            # Calculate contributions using treeinterpreter
            prediction, bias, contributions = ti.predict(model, X)
            
            # Process the contributions
            results = self._process_importance_values(contributions, 'treeinterpreter')
            
            # Add additional treeinterpreter-specific data
            results['bias'] = bias
            results['prediction'] = prediction
            
            return results
        except AttributeError as ae:
            if "'LGBMClassifier' object has no attribute 'n_outputs_'" in str(ae):
                print("LGBMClassifier model is missing n_outputs_ attribute required by treeinterpreter")
                print("Falling back to SHAP...")
                return self._calculate_shap_importance(model, X)
            else:
                print(f"Error in treeinterpreter calculation: {str(ae)}")
                print("Falling back to SHAP...")
                return self._calculate_shap_importance(model, X)
        except ValueError as ve:
            if "Wrong model type" in str(ve):
                print("Treeinterpreter does not support this model type. It requires DecisionTreeClassifier or DecisionTreeRegressor.")
                print("Falling back to SHAP...")
                return self._calculate_shap_importance(model, X)
            else:
                print(f"Error in treeinterpreter calculation: {str(ve)}")
                print("Falling back to SHAP...")
                return self._calculate_shap_importance(model, X)
        except Exception as e:
            print(f"Error in treeinterpreter calculation: {str(e)}")
            print("Falling back to SHAP...")
            return self._calculate_shap_importance(model, X)
    
    def _convert_lightgbm_to_sklearn(self, model: Any, X: np.ndarray) -> Any:
        """
        Convert a LightGBM Booster model to a sklearn-compatible model.
        
        Args:
            model: LightGBM Booster model
            X: Input features used to fit the model
            
        Returns:
            Sklearn-compatible LightGBM model
        """
        import lightgbm as lgbm
        
        # Use a small subset of data to create a sklearn model
        subset_size = min(1000, X.shape[0])
        X_subset = X[:subset_size]
        
        # Create a prediction target for fitting
        y_pred = model.predict(X_subset)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Create a sklearn-compatible model
        sklearn_model = lgbm.LGBMClassifier()
        
        # Set the params from the original model when possible
        if hasattr(model, 'params'):
            params = model.params.copy()
            if 'num_iterations' in params:
                params['n_estimators'] = int(params.pop('num_iterations'))
            sklearn_model.set_params(**{k: v for k, v in params.items() 
                                       if k in sklearn_model.get_params()})
        
        # Fit the model with X_subset and predicted y
        sklearn_model.fit(X_subset, y_pred_binary)
        
        # Add n_outputs_ attribute for treeinterpreter compatibility
        if not hasattr(sklearn_model, 'n_outputs_'):
            # Binary classification has 1 output
            sklearn_model.n_outputs_ = 1
        
        print("Successfully converted LightGBM model to sklearn-compatible model")
        
        return sklearn_model
    
    def save_importance_results(self, file_path: str = None) -> str:
        """
        Save the feature importance results to a CSV file.
        
        Args:
            file_path: Path to save the results to (default: None)
            
        Returns:
            Path to the saved results file
        """
        if file_path is None:
            # Create a default file path
            target_var = self.target_variable
            file_path = os.path.join(self.output_dir, f"{target_var}_{self.importance_method}_feature_importance.csv")
        
        if self.importance_df is not None:
            # Save to CSV
            self.importance_df.to_csv(file_path, index=False)
            print(f"Feature importance results saved to {file_path}")
        else:
            raise ValueError("No feature importance results available.")
        
        return file_path
    
    def save_importance_values(self, file_path: str = None) -> str:
        """
        Save the raw feature importance values to a pickle file.
        
        Args:
            file_path: Path to save the values to (default: None)
            
        Returns:
            Path to the saved values file
        """
        if file_path is None:
            # Create a default file path
            target_var = self.target_variable
            file_path = os.path.join(self.output_dir, f"{target_var}_{self.importance_method}_importance_values.pkl")
        
        if self.importance_values is not None:
            # Save to pickle
            with open(file_path, 'wb') as f:
                pickle.dump(self.importance_values, f)
            print(f"Feature importance values saved to {file_path}")
        else:
            raise ValueError("No feature importance values available.")
        
        return file_path
    
    def plot_feature_importance(self, 
                              n_features: int = 20, 
                              figsize: Tuple[int, int] = (12, 8),
                              save_path: str = None) -> None:
        """
        Plot feature importance.
        
        Args:
            n_features: Number of top features to plot (default: 20)
            figsize: Figure size (default: (12, 8))
            save_path: Path to save the plot to (default: None)
        """
        if self.importance_df is None:
            raise ValueError("No feature importance results available.")
        
        # Get top N features
        top_features = self.importance_df.head(n_features)
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Plot horizontal bar chart
        plt.barh(top_features['feature'], top_features['importance'])
        
        # Add labels
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {n_features} Features by {self.importance_method} Importance')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save if path is provided
        if save_path is not None:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to {save_path}")
        
        # Show plot
        plt.show()
    
    def get_importance_df(self) -> pd.DataFrame:
        """
        Get the feature importance DataFrame.
        
        Returns:
            DataFrame containing feature importance results
        """
        return self.importance_df
    
    def get_importance_values(self) -> np.ndarray:
        """
        Get the raw feature importance values.
        
        Returns:
            Array containing feature importance values
        """
        return self.importance_values
    
    def get_feature_rankings(self) -> pd.DataFrame:
        """
        Get the feature rankings.
        
        Returns:
            DataFrame containing feature rankings
        """
        return self.feature_importance_rankings 