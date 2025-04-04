# LGBM Classification ML Pipeline

This project implements a machine learning pipeline for LGBM classification models. The pipeline handles data loading, validation, preprocessing, model training with hyperparameter optimization, and feature importance analysis.

## Project Structure

- `ml_pipeline/`: Core pipeline modules
  - `data_loader.py`: DataLoader class for loading and validating data
  - `data_processor.py`: DataProcessor class for preprocessing data
  - `lgbm_model.py`: LGBMModel class for model training with Optuna hyperparameter tuning
  - `feature_importance.py`: FeatureImportance class for calculating feature importance using SHAP or treeinterpreter
  - `ml_pipeline.py`: ML_Pipeline class for integrating the components
  - `tests/`: Test scripts for each component
- `ml_pipeline_demo.ipynb`: Jupyter notebook demonstrating the basic pipeline
- `model_training_demo.ipynb`: Jupyter notebook demonstrating model training and feature importance analysis
- `data_config_train.json`: Configuration for training
- `data_config_test.json`: Configuration for testing
- `model_config.json`: Model configuration
- `feature_metadata.xlsx`: Feature metadata

## Installation

1. Ensure you have Python 3.10 installed
2. Create a virtual environment:
   ```
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Configuration Files

### Model Config

The `model_config.json` file specifies how the model should be trained:

```json
{
  "target_variable": "target",
  "id_variables": ["id_col", "date_col"],
  "problem_type": "classification",
  "evaluation_metric": "rocauc",
  "better_performance": "gt",
  "models": ["lgbm"],
  "k_folds": 5,
  "retrain_with_whole_dataset": true,
  "sample_for_contribution": 1.0,
  "importance_extraction_method": "treeinterpreter",
  "num_hp_searches": 5,
  "hyperparameter_space": {
    "lgbm": {
      "boosting_type": "trial.suggest_categorical('boosting_type', ['gbdt'])",
      "learning_rate": "trial.suggest_loguniform('learning_rate', 0.01, 0.2)",
      "num_leaves": "int(trial.suggest_discrete_uniform('num_leaves', 30, 150, 1))",
      "feature_fraction": "trial.suggest_uniform('feature_fraction', 0.1, 1.0)",
      "reg_lambda": "trial.suggest_uniform('reg_lambda', 0.0, 0.1)",
      "n_estimators": "int(trial.suggest_discrete_uniform('n_estimators', 100, 200, 1))",
      "class_weight": "trial.suggest_categorical('class_weight', [None, 'balanced'])",
      "min_data_in_leaf": "int(trial.suggest_discrete_uniform('min_data_in_leaf', 10, 1000, 5))"
    }
  }
}
```

### Data Config (Training)

The `data_config_train.json` file specifies the data for training:

```json
{
  "training": true,
  "metadata_file_path": "feature_metadata.xlsx",
  "input_data_source": "CSV",
  "file_path": "example_inputs_v1.csv"
}
```

### Data Config (Testing)

The `data_config_test.json` file specifies the data for testing:

```json
{
  "training": false,
  "metadata_file_path": "feature_metadata.xlsx",
  "input_data_source": "CSV",
  "file_path": "example_inputs_test_v1.csv",
  "data_processor_file_path": "trained_model_outputs_path/target_classification_lgbm_data_processor.pkl",
  "model_file_path": "trained_model_outputs_path/target_classification_lgbm.pkl",
  "predictions_output_path": "trained_model_outputs_path/classification_v1.csv"
}
```

### Feature Metadata

The `feature_metadata.xlsx` file specifies information about each feature, including:
- Type (categorical, float, boolean)
- Whether to use it in the model (0 or 1)
- Imputation method for missing values
- Transformation method (standardization, one-hot encoding, etc.)
- Super-category grouping

## Usage

### Basic Usage

```python
from ml_pipeline import ML_Pipeline

# For training
pipeline = ML_Pipeline(modeling_config="model_config.json", data_config="data_config_train.json")
pipeline.run()

# For testing
pipeline = ML_Pipeline(modeling_config="model_config.json", data_config="data_config_test.json")
pipeline.run()
```

### Model Training with Hyperparameter Tuning

```python
from ml_pipeline import DataLoader, DataProcessor, LGBMModel

# Load and preprocess data
data_loader = DataLoader("data_config_train.json", "model_config.json")
model_config = data_loader.get_model_config()
feature_metadata = data_loader.get_feature_metadata()
data = data_loader.get_data()

data_processor = DataProcessor(feature_metadata, model_config)
X, y = data_processor.fit_transform_full_dataset(data)

# Train model with hyperparameter tuning
lgbm_model = LGBMModel(model_config)
lgbm_model.train_with_full_dataset(X, y)

# Save the trained model
model_path = lgbm_model.save_model()
```

### Feature Importance Analysis

```python
from ml_pipeline import FeatureImportance

# Get feature names
feature_names = [...] # List of feature names

# Calculate SHAP importance
shap_importance = FeatureImportance(
    model_config=model_config,
    feature_names=feature_names,
    output_dir="output_directory"
)

shap_results = shap_importance.calculate_importance(model, X)
importance_df = shap_importance.get_importance_df()

# Plot and save results
shap_importance.plot_feature_importance(n_features=10)
shap_importance.save_importance_results()
```

### Running Tests

To run the component tests:

```
python -m ml_pipeline.tests.test_data_loader
python -m ml_pipeline.tests.test_data_processor
python -m ml_pipeline.tests.test_lgbm_model
python -m ml_pipeline.tests.test_feature_importance
python -m ml_pipeline.tests.test_ml_pipeline
```

### Jupyter Notebooks

For interactive demonstrations, run the Jupyter notebooks:

```
jupyter notebook ml_pipeline_demo.ipynb     # Basic pipeline demo
jupyter notebook model_training_demo.ipynb  # Model training and feature importance demo
```

## Pipeline Features

1. **Data Loading & Validation**
   - Validates configuration files
   - Loads and validates data against metadata
   - Converts data to appropriate types

2. **Data Processing**
   - Performs k-fold cross-validation splitting
   - Applies feature-specific transformations:
     - Standardizes numerical features
     - Encodes categorical features
     - Converts binary features
     - Handles missing values according to specifications
   - Preserves ID columns for linking predictions

3. **Model Training with Hyperparameter Optimization**
   - Uses Optuna for hyperparameter tuning
   - Supports various evaluation metrics (AUC, precision, recall, accuracy)
   - Optimizes for either higher or lower metric values
   - Records hyperparameter search history
   - Can retrain on the full dataset with optimal hyperparameters

4. **Feature Importance Analysis**
   - Calculates feature importance using SHAP or treeinterpreter
   - Supports sampling a subset of data for faster computation
   - Generates feature importance rankings
   - Provides visualization capabilities
   - Saves feature importance results for further analysis

5. **Prediction**
   - Loads trained model and preprocessor
   - Processes test data
   - Generates predictions
   - Adds ID columns back to predictions
   - Saves predictions to specified path