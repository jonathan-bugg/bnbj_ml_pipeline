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
    Class for preprocessing data for LGBM models.
    
    This class prepares pandas dataframes for training in LGBM models.
    For training, it splits data into k-fold cross validation sets.
    For testing, it uses a saved data transformer to preprocess the test data.
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
            output_dir: Directory to save outputs to (default: 'trained_model_outputs_path')
        """
        self.feature_metadata = feature_metadata
        self.model_config = model_config
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Store preprocessing attributes
        self.id_columns = model_config.get("id_variables", [])
        self.target_column = model_config.get("target_variable", "")
        self.k_folds = model_config.get("k_folds", 5)
        self.problem_type = model_config.get("problem_type", "classification")
        
        # Extract feature information from metadata
        self.numerical_features = []
        self.categorical_features = []
        self.binary_features = []
        self.one_hot_features = []
        self.leave_as_na_features = []
        
        self._prepare_feature_lists()
        
        # Preprocessing pipeline
        self.preprocessor = None
        
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
        
        # Numerical features with missing values preserved
        numerical_na_cols = [col for col in self.numerical_features if col in self.leave_as_na_features]
        if numerical_na_cols:
            # We will handle these by not transforming them
            pass
        
        # Categorical features (not one-hot encoded)
        cat_cols = [col for col in self.categorical_features if col not in self.leave_as_na_features]
        if cat_cols:
            cat_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent'))
            ])
            transformers.append(('cat', cat_transformer, cat_cols))
        
        # Categorical features with missing values preserved
        cat_na_cols = [col for col in self.categorical_features if col in self.leave_as_na_features]
        if cat_na_cols:
            # We will handle these by not transforming them
            pass
        
        # One-hot encoded features
        onehot_cols = [col for col in self.one_hot_features if col not in self.leave_as_na_features]
        if onehot_cols:
            onehot_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('onehot', onehot_transformer, onehot_cols))
        
        # One-hot encoded features with missing values preserved
        onehot_na_cols = [col for col in self.one_hot_features if col in self.leave_as_na_features]
        if onehot_na_cols:
            onehot_na_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
            ])
            transformers.append(('onehot_na', onehot_na_transformer, onehot_na_cols))
        
        # Binary features
        bin_cols = [col for col in self.binary_features if col not in self.leave_as_na_features]
        if bin_cols:
            # For binary features, we will simply impute with 0 (unless otherwise specified)
            bin_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value=0))
            ])
            transformers.append(('bin', bin_transformer, bin_cols))
        
        # Binary features with missing values preserved
        bin_na_cols = [col for col in self.binary_features if col in self.leave_as_na_features]
        if bin_na_cols:
            # We will handle these by not transforming them
            pass
        
        # Create the column transformer
        return ColumnTransformer(
            transformers=transformers,
            remainder='passthrough'
        )
    
    def _prepare_data_for_training(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the data for training by removing ID columns.
        
        Args:
            data: DataFrame containing the input data
            
        Returns:
            Tuple containing X (features) and y (target)
        """
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Extract the target variable
        y = df[self.target_column]
        
        # Remove ID columns and target variable from features
        X = df.drop(columns=self.id_columns + [self.target_column])
        
        return X, y
    
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
                # Convert binary columns to numeric (0/1)
                X_copy[col] = X_copy[col].astype(float)
                
        return X_copy
    
    def create_folds(self, data: pd.DataFrame) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create k-fold cross-validation indices.
        
        Args:
            data: DataFrame containing the input data
            
        Returns:
            List of tuples containing train and validation indices for each fold
        """
        X, y = self._prepare_data_for_training(data)
        
        # Choose the right K-Fold class based on the problem type
        if self.problem_type == "classification":
            kf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=42)
            fold_indices = list(kf.split(X, y))
        else:
            kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
            fold_indices = list(kf.split(X))
        
        return fold_indices
    
    def preprocess_training_data(self, data: pd.DataFrame, fold_indices: List[Tuple[np.ndarray, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Preprocess the training data using k-fold cross-validation.
        
        Args:
            data: DataFrame containing the input data
            fold_indices: List of tuples containing train and validation indices for each fold
            
        Returns:
            Dictionary containing the preprocessed data for each fold
        """
        X, y = self._prepare_data_for_training(data)
        X = self._handle_binary_features(X)
        
        # If fold indices are not provided, create them
        if fold_indices is None:
            fold_indices = self.create_folds(data)
        
        # Store preprocessed data for each fold
        preprocessed_data = {
            'original_data': data,
            'X': X,
            'y': y,
            'folds': []
        }
        
        # For each fold, fit the preprocessor on the training data and transform both train and validation data
        for fold_idx, (train_idx, val_idx) in enumerate(fold_indices):
            # Initialize the preprocessor
            self.preprocessor = self._build_preprocessor()
            
            # Get train and validation data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Convert binary features to numeric for both train and validation sets
            X_train = self._handle_binary_features(X_train)
            X_val = self._handle_binary_features(X_val)
            
            # Fit the preprocessor on the training data
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
            
            preprocessed_data['folds'].append(fold_data)
        
        return preprocessed_data
    
    def fit_transform_full_dataset(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the preprocessor on the full dataset and transform it.
        
        Args:
            data: DataFrame containing the input data
            
        Returns:
            Tuple containing the transformed X and y
        """
        X, y = self._prepare_data_for_training(data)
        X = self._handle_binary_features(X)
        
        # Initialize the preprocessor
        self.preprocessor = self._build_preprocessor()
        
        # Fit and transform the full dataset
        X_transformed = self.preprocessor.fit_transform(X)
        
        # Save the fitted preprocessor
        self.save_preprocessor()
        
        return X_transformed, y
    
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
            with open(file_path, 'wb') as f:
                pickle.dump(self.preprocessor, f)
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
            self.preprocessor = pickle.load(f)
    
    def preprocess_test_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the test data using a previously fitted preprocessor.
        
        Args:
            data: DataFrame containing the test data
            
        Returns:
            Preprocessed test data
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor has not been fitted or loaded yet.")
        
        # Create a copy to avoid modifying the original data
        df = data.copy()
        
        # Extract ID columns before preprocessing
        id_data = {col: df[col] for col in self.id_columns if col in df.columns}
        
        # Remove ID columns from features
        X = df.drop(columns=[col for col in self.id_columns if col in df.columns])
        
        # Handle binary features
        X = self._handle_binary_features(X)
        
        # Transform the data
        X_transformed = self.preprocessor.transform(X)
        
        return X_transformed, id_data
    
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