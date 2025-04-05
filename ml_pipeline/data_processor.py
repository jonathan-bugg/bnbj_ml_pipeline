import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


class DataProcessor:
    """
    Class for preprocessing data for ML models.
    
    This class prepares pandas dataframes for model training and inference.
    """
    
    def __init__(self, 
                 feature_metadata: pd.DataFrame, 
                 model_config: Dict[str, Any],
                 output_dir: str = 'trained_model_outputs_path'):
        """
        Initialize the DataProcessor.
        
        Args:
            feature_metadata: DataFrame containing the feature metadata
            model_config: Dictionary containing the model configuration
            output_dir: Directory to save outputs to
        """
        self.feature_metadata = feature_metadata
        self.model_config = model_config
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract configuration
        self.id_columns = model_config.get("id_variables", [])
        self.target_column = model_config.get("target_variable", "")
        self.k_folds = model_config.get("k_folds", 5)
        self.problem_type = model_config.get("problem_type", "classification")
        
        # Initialize feature lists
        self.numerical_features = []
        self.categorical_features = []
        self.binary_features = []
        self.one_hot_features = []
        self.leave_as_na_features = []
        self._prepare_feature_lists()
        
        # Preprocessing pipeline
        self.preprocessor = None
        
        # Store mappings for categorical features
        self.categorical_mappings = {}
        
    def _prepare_feature_lists(self) -> None:
        """
        Prepare lists of features by type and transformation method.
        """
        for _, feature in self.feature_metadata.iterrows():
            feature_name = feature["feature_name"]
            feature_type = feature["type"]
            feature_use = feature["use"]
            feature_imputation = feature["imputation"]
            feature_transformation = feature["transformation"]
            
            # Skip if not used in model or if it's an ID column or target
            if feature_use != 1 or feature_name in self.id_columns or feature_name == self.target_column:
                continue
            
            # Categorize features based on type and transformation
            if feature_type in ["float", "number"]:
                self.numerical_features.append(feature_name)
                
                # Check if missing values should be left as NA
                if feature_imputation.lower() in ["leave as n/a", "unknown", "none", "nan", ""]:
                    self.leave_as_na_features.append(feature_name)
                    
            elif feature_type == "category":
                # Check transformation method
                if feature_transformation and feature_transformation.lower() == "one-hot":
                    self.one_hot_features.append(feature_name)
                else:
                    self.categorical_features.append(feature_name)
                    
                # Check if missing values should be left as NA
                if feature_imputation.lower() in ["leave as n/a", "unknown", "none", "nan", ""]:
                    self.leave_as_na_features.append(feature_name)
                    
            elif feature_type == "bool":
                self.binary_features.append(feature_name)
                
                # Check if missing values should be left as NA
                if feature_imputation.lower() in ["leave as n/a", "unknown", "none", "nan", ""]:
                    self.leave_as_na_features.append(feature_name)
    
    def _build_preprocessor(self) -> ColumnTransformer:
        """
        Build the preprocessing pipeline based on feature types and transformations.
        
        Returns:
            ColumnTransformer object for preprocessing the data
        """
        transformers = []
        
        # Numerical features preprocessing
        numerical_cols = [col for col in self.numerical_features if col not in self.leave_as_na_features]
        if numerical_cols:
            num_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ])
            transformers.append(('num', num_transformer, numerical_cols))
        
        # Categorical features (not one-hot encoded)
        cat_cols = [col for col in self.categorical_features if col not in self.leave_as_na_features]
        if cat_cols:
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ])
            transformers.append(('cat', cat_transformer, cat_cols))
        
        # One-hot encoded features
        onehot_cols = [col for col in self.one_hot_features if col not in self.leave_as_na_features]
        if onehot_cols:
            onehot_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
            ])
            transformers.append(('onehot', onehot_transformer, onehot_cols))
        
        # One-hot encoded features with missing values preserved
        onehot_na_cols = [col for col in self.one_hot_features if col in self.leave_as_na_features]
        if onehot_na_cols:
            onehot_na_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(sparse=False, handle_unknown='ignore'))
            ])
            transformers.append(('onehot_na', onehot_na_transformer, onehot_na_cols))
        
        # Binary features
        bin_cols = [col for col in self.binary_features if col not in self.leave_as_na_features]
        if bin_cols:
            bin_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            ])
            transformers.append(('bin', bin_transformer, bin_cols))
        
        # Create the column transformer
        return ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        ).set_output(transform='pandas')
    
    def _handle_non_numeric_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle non-numeric values in all columns by encoding or converting them.
        
        Args:
            X: DataFrame containing the features
            
        Returns:
            DataFrame with non-numeric values converted to numeric
        """
        X_copy = X.copy()
        
        # Process all columns
        for col in X_copy.columns:
            # Check if column has non-numeric values
            try:
                X_copy[col] = pd.to_numeric(X_copy[col], errors='raise')
            except ValueError:
                print(f"Error converting data to numeric: could not convert string to float in column {col}")
                # Handle non-numeric values by encoding them
                if X_copy[col].dtype == 'object':
                    # Create a mapping dictionary if not already present
                    if col not in self.categorical_mappings:
                        unique_values = X_copy[col].astype(str).unique()
                        self.categorical_mappings[col] = {val: i for i, val in enumerate(unique_values)}
                    
                    # Apply mapping
                    X_copy[col] = X_copy[col].astype(str).map(self.categorical_mappings[col])
                    print(f"Column {col} contains non-numeric values. Converting to numeric encoding.")
        
        return X_copy
    
    def _handle_binary_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Handle binary features by converting them to numeric values.
        
        Args:
            X: DataFrame containing the features
            
        Returns:
            DataFrame with binary features converted to numeric
        """
        X_copy = X.copy()
        
        for col in self.binary_features:
            if col in X_copy.columns:
                try:
                    X_copy[col] = X_copy[col].astype(float)
                except ValueError:
                    # Let the more comprehensive handler take care of this
                    pass
                
        return X_copy
    
    def process_data(self, data: pd.DataFrame, is_training: bool = True, create_folds: bool = True) -> Dict[str, Any]:
        """
        Process data for training or inference.
        
        Args:
            data: DataFrame containing the input data
            is_training: Whether this is for training (True) or inference (False)
            create_folds: Whether to create cross-validation folds (True for CV, False for full dataset training)
            
        Returns:
            Dictionary containing the processed data
        """
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Extract target and features
        if is_training:
            y = df[self.target_column]
            X = df.drop(columns=self.id_columns + [self.target_column])
        else:
            # For inference, extract ID columns and no target
            id_data = {col: df[col] for col in self.id_columns if col in df.columns}
            X = df.drop(columns=[col for col in self.id_columns if col in df.columns])
            y = None
        
        # Handle non-numeric values
        # X = self._handle_non_numeric_values(X)
        
        # Handle binary features
        X = self._handle_binary_features(X)
        
        # # Verify all values are numeric
        # for col in X.columns:
        #     try:
        #         X[col] = X[col].astype(float)
        #     except Exception as e:
        #         print(f"WARNING: Column {col} still has non-numeric values after preprocessing: {str(e)}")
        #         # Force conversion for problematic columns
        #         X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                
        # print(f"Data converted to numeric after fixing: X shape: {X.shape}, dtypes: {X.dtypes}")
        
        # Initialize result dictionary
        result = {
            'X': X,
            'y': y
        }
        
        if not is_training:
            # For inference, transform the data using a previously fit preprocessor
            if self.preprocessor is None:
                raise ValueError("Preprocessor has not been fitted or loaded yet.")
            
            X_transformed = self.preprocessor.transform(X)
            result['X_transformed'] = X_transformed
            result['id_data'] = id_data
            return result
            
        # For training mode
        if create_folds:
            # Create fold indices for cross-validation
            if self.problem_type == "classification":
                kf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
                fold_indices = list(kf.split(X, y))
            else:
                kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
                fold_indices = list(kf.split(X))
                
            # Store preprocessed data for each fold
            folds = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
                # Get train and validation data
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Initialize and fit the preprocessor on the training data
                self.preprocessor = self._build_preprocessor()
                X_train_transformed = self.preprocessor.fit_transform(X_train)
                
                # Transform the validation data
                X_val_transformed = self.preprocessor.transform(X_val)
                
                # Store the fold data
                fold_data = {
                    'fold_idx': fold_idx,
                    'train_idx': train_idx,
                    'val_idx': val_idx,
                    'X_train': X_train,
                    'y_train': y_train,
                    'X_val': X_val,
                    'y_val': y_val,
                    'X_train_transformed': X_train_transformed,
                    'X_val_transformed': X_val_transformed,
                    'preprocessor': self.preprocessor
                }
                
                folds.append(fold_data)
                
            result['folds'] = folds
        else:
            # For full dataset training, fit the preprocessor on the entire dataset
            self.preprocessor = self._build_preprocessor()
            X_transformed = self.preprocessor.fit_transform(X)
            result['X_transformed'] = X_transformed
            
            # Save the fitted preprocessor
            self.save_preprocessor()
            
        return result
    
    def save_preprocessor(self, file_path: str = None) -> str:
        """
        Save the fitted preprocessor to a file.
        
        Args:
            file_path: Path to save the preprocessor to (default: None)
            
        Returns:
            Path to the saved preprocessor file
        """
        if file_path is None:
            # Create a default file path
            target_var = self.target_column
            problem_type = self.problem_type
            model_type = self.model_config.get("models", ["lgbm"])[0]
            file_path = os.path.join(self.output_dir, f"{target_var}_{problem_type}_{model_type}_data_processor.pkl")
        
        if self.preprocessor is not None:
            # Save preprocessor and categorical mappings together
            to_save = {
                'preprocessor': self.preprocessor,
                'categorical_mappings': self.categorical_mappings
            }
            with open(file_path, 'wb') as f:
                pickle.dump(to_save, f)
        else:
            raise ValueError("Preprocessor has not been fitted yet.")
        
        return file_path
    
    def load_preprocessor(self, file_path: str) -> None:
        """
        Load a fitted preprocessor from a file.
        
        Args:
            file_path: Path to the saved preprocessor file
        """
        with open(file_path, 'rb') as f:
            saved_data = pickle.load(f)
            
        # Handle both old and new format
        if isinstance(saved_data, dict):
            self.preprocessor = saved_data.get('preprocessor')
            self.categorical_mappings = saved_data.get('categorical_mappings', {})
        else:
            self.preprocessor = saved_data
            self.categorical_mappings = {}
    
    def add_ids_to_predictions(self, predictions: np.ndarray, id_data: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Add ID columns back to the predictions.
        
        Args:
            predictions: Array of predictions
            id_data: Dictionary containing ID columns
            
        Returns:
            DataFrame containing predictions with ID columns
        """
        # Create a dataframe from predictions
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            # For multiclass problems
            pred_cols = [f"prob_class_{i}" for i in range(predictions.shape[1])]
            pred_df = pd.DataFrame(predictions, columns=pred_cols)
            pred_df['prediction'] = predictions.argmax(axis=1)
        else:
            # For binary classification or regression
            pred_df = pd.DataFrame({'prediction': predictions.flatten()})
        
        # Add ID columns
        for col, values in id_data.items():
            pred_df[col] = values.values
        
        return pred_df 