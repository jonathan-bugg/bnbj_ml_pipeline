import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any, Optional


class DataLoader:
    """
    Class for loading and validating data based on configuration files.
    
    This class reads in the data config and modeling config, validates them,
    and ensures that the input data for either training or testing is valid
    and matches the feature_metadata.xlsx specifications.
    """
    
    def __init__(self, data_config_path: str, model_config_path: str):
        """
        Initialize the DataLoader with paths to configuration files.
        
        Args:
            data_config_path: Path to the data configuration JSON file
            model_config_path: Path to the model configuration JSON file
        """
        self.data_config_path = data_config_path
        self.model_config_path = model_config_path
        
        # Load configurations
        self.data_config = self._load_json_config(data_config_path)
        self.model_config = self._load_json_config(model_config_path)
        
        # Validate configurations
        self._validate_config_files()
        
        # Load feature metadata
        self.metadata_path = self.data_config.get("metadata_file_path", "")
        if self.metadata_path.startswith("/"):
            # Handle relative path starting with slash
            self.metadata_path = self.metadata_path[1:]
        self.feature_metadata = self._load_feature_metadata()
        
        # Load input data
        self.input_data = self._load_input_data()
        
        # Validate data against metadata
        self._validate_data_against_metadata()
        
    def _load_json_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load and parse a JSON configuration file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            Dictionary containing the configuration
            
        Raises:
            FileNotFoundError: If the configuration file doesn't exist
            ValueError: If the configuration file isn't valid JSON
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file {config_path} not found.")
        except json.JSONDecodeError:
            raise ValueError(f"Configuration file {config_path} is not valid JSON.")
    
    def _validate_config_files(self) -> None:
        """
        Validate that the configuration files contain the required fields.
        
        Raises:
            ValueError: If a required field is missing from a configuration file
        """
        # Check data config required fields
        required_data_fields = ["training", "metadata_file_path", "input_data_source", "file_path"]
        for field in required_data_fields:
            if field not in self.data_config:
                raise ValueError(f"Required field '{field}' missing from data config file.")
        
        # If it's a testing config, additional fields are required
        if not self.data_config["training"]:
            test_required_fields = ["data_processor_file_path", "model_file_path", "predictions_output_path"]
            for field in test_required_fields:
                if field not in self.data_config:
                    raise ValueError(f"Required field '{field}' missing from testing data config file.")
        
        # Check model config required fields
        required_model_fields = ["target_variable", "id_variables", "problem_type", 
                               "evaluation_metric", "better_performance", "models", 
                               "k_folds", "retrain_with_whole_dataset"]
        for field in required_model_fields:
            if field not in self.model_config:
                raise ValueError(f"Required field '{field}' missing from model config file.")
        
        # Validate that the model type is supported
        supported_models = ["lgbm"]
        for model in self.model_config["models"]:
            if model not in supported_models:
                raise ValueError(f"Unsupported model type '{model}'. Supported types are: {', '.join(supported_models)}")
        
        # Validate problem type
        if self.model_config["problem_type"] not in ["classification", "regression"]:
            raise ValueError("Problem type must be 'classification' or 'regression'.")
    
    def _load_feature_metadata(self) -> pd.DataFrame:
        """
        Load and parse the feature metadata Excel file.
        
        Returns:
            DataFrame containing the feature metadata
            
        Raises:
            FileNotFoundError: If the metadata file doesn't exist
            ValueError: If the metadata file doesn't have the expected structure
        """
        try:
            metadata_df = pd.read_excel(self.metadata_path)
            
            # Transform metadata_df to make it more usable
            # First row contains column types
            feature_types = metadata_df.iloc[0].to_dict()
            # Second row contains usage flags
            feature_use = metadata_df.iloc[1].to_dict()
            # Third row contains imputation methods
            feature_imputation = metadata_df.iloc[2].to_dict()
            # Fourth row contains category transformations
            feature_transformation = metadata_df.iloc[3].to_dict()
            # Fifth row contains super categories
            feature_super_category = metadata_df.iloc[4].to_dict()
            
            # Create a clean DataFrame for easier access
            feature_info = []
            for col in metadata_df.columns:
                if col != "Unnamed: 0":
                    feature_info.append({
                        "feature_name": col,
                        "type": feature_types.get(col, ""),
                        "use": feature_use.get(col, 0),
                        "imputation": feature_imputation.get(col, ""),
                        "transformation": feature_transformation.get(col, ""),
                        "super_category": feature_super_category.get(col, "")
                    })
            
            return pd.DataFrame(feature_info)
        
        except FileNotFoundError:
            raise FileNotFoundError(f"Feature metadata file {self.metadata_path} not found.")
        except Exception as e:
            raise ValueError(f"Error parsing feature metadata file: {str(e)}")
    
    def _load_input_data(self) -> pd.DataFrame:
        """
        Load input data from the specified source.
        
        Currently supports CSV files.
        
        Returns:
            DataFrame containing the input data
            
        Raises:
            ValueError: If the data source is not supported or the file doesn't exist
        """
        data_source = self.data_config["input_data_source"]
        file_path = self.data_config["file_path"]
        
        if data_source == "CSV":
            try:
                return pd.read_csv(file_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Input data file {file_path} not found.")
            except Exception as e:
                raise ValueError(f"Error loading input data: {str(e)}")
        else:
            raise ValueError(f"Unsupported data source type: {data_source}")
    
    def _validate_data_against_metadata(self) -> None:
        """
        Validate that the input data matches the feature metadata.
        
        Checks that all required columns are present and have the expected data types.
        
        Raises:
            ValueError: If the data doesn't match the metadata
        """
        # Check that all columns in the metadata are present in the data
        for _, feature in self.feature_metadata.iterrows():
            feature_name = feature["feature_name"]
            
            # Skip validation for columns that are not used in the model
            if feature["use"] != 1:
                continue
                
            # Check if the feature exists in the input data
            if feature_name not in self.input_data.columns:
                raise ValueError(f"Feature '{feature_name}' specified in metadata is missing from input data.")
        
        # Check that target variable is present in training data
        if self.data_config["training"]:
            target_var = self.model_config["target_variable"]
            if target_var not in self.input_data.columns:
                raise ValueError(f"Target variable '{target_var}' is missing from training data.")
    
    def get_data_config(self) -> Dict[str, Any]:
        """
        Get the data configuration.
        
        Returns:
            Dictionary containing the data configuration
        """
        return self.data_config
    
    def get_model_config(self) -> Dict[str, Any]:
        """
        Get the model configuration.
        
        Returns:
            Dictionary containing the model configuration
        """
        return self.model_config
    
    def get_feature_metadata(self) -> pd.DataFrame:
        """
        Get the feature metadata.
        
        Returns:
            DataFrame containing the feature metadata
        """
        return self.feature_metadata
    
    def get_data(self) -> pd.DataFrame:
        """
        Get the loaded and validated input data with correct data types.
        
        Returns:
            DataFrame containing the input data with correct data types
        """
        # Create a copy to avoid modifying the original data
        data = self.input_data.copy()
        
        # Convert data types based on metadata
        for _, feature in self.feature_metadata.iterrows():
            feature_name = feature["feature_name"]
            feature_type = feature["type"]
            
            # Skip if feature is not in the data
            if feature_name not in data.columns:
                continue
            
            # Convert to the correct data type
            if feature_type == "category":
                # Convert to string first to handle NaN values correctly
                data[feature_name] = data[feature_name].astype(str)
                data[feature_name] = data[feature_name].replace('nan', np.nan)
                data[feature_name] = data[feature_name].astype('category')
            elif feature_type == "bool":
                # Handle various boolean representations
                if data[feature_name].dtype != bool:
                    data[feature_name] = data[feature_name].map({
                        'True': True, 'true': True, '1': True, 1: True, 
                        'False': False, 'false': False, '0': False, 0: False
                    }, na_action='ignore')
                    data[feature_name] = data[feature_name].astype(bool)
            elif feature_type in ["float", "number"]:
                # Convert to float, ignoring errors for now (they'll be handled in preprocessing)
                data[feature_name] = pd.to_numeric(data[feature_name], errors='coerce')
        
        return data 