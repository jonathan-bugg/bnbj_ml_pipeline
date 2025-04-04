# LGBM Classification ML Pipeline

This project implements a machine learning pipeline for LGBM classification models. The pipeline handles data loading, validation, preprocessing, model training, and prediction.

## Project Structure

- `ml_pipeline/`: Core pipeline modules
  - `data_loader.py`: DataLoader class for loading and validating data
  - `data_processor.py`: DataProcessor class for preprocessing data
  - `ml_pipeline.py`: ML_Pipeline class for integrating the components
  - `tests/`: Test scripts for each component
- `ml_pipeline_demo.ipynb`: Jupyter notebook demonstrating the pipeline
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
  "retrain_with_whole_dataset": true
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

### Running Tests

To run the component tests:

```
python -m ml_pipeline.tests.test_data_loader
python -m ml_pipeline.tests.test_data_processor
python -m ml_pipeline.tests.test_ml_pipeline
```

### Jupyter Notebook

For an interactive demonstration, run the Jupyter notebook:

```
jupyter notebook ml_pipeline_demo.ipynb
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

3. **Model Training**
   - Trains LGBM models on each fold
   - Evaluates performance using specified metrics
   - Retrains on full dataset if specified
   - Saves trained model and preprocessor

4. **Prediction**
   - Loads trained model and preprocessor
   - Processes test data
   - Generates predictions
   - Adds ID columns back to predictions
   - Saves predictions to specified path