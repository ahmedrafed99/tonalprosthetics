from kedro.pipeline import Pipeline, node
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from kedro.extras.datasets.pickle import PickleDataSet
from nodes import preprocess_data, split_data, create_model, train_model, evaluate_model

def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            node(
                func=preprocess_data,
                inputs="tonal_exams",
                outputs="preprocessed_data",
                name="preprocess_data_node"
            ),
            node(
                func=split_data,
                inputs=dict(data="preprocessed_data", 
                            test_size="params:test_size", 
                            random_state="params:random_state"),
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node"
            ),
            node(
                func=create_model,
                inputs=dict(units="params:units",
                            activation="params:activation",
                            l2_value="params:l2_value",
                            dropout_rate="params:dropout_rate",
                            learning_rate="params:learning_rate"),
                outputs="model",
                name="create_model_node"
            ),
            node(
                func=train_model,
                inputs=["X_train", "y_train", "model", "params:epochs"],
                outputs="trained_model",
                name="train_model_node"
            ),
            node(
                func=evaluate_model,
                inputs=["X_test", "y_test", "trained_model"],
                outputs="evaluation_results",
                name="evaluate_model_node"
            
            ),
        ]
    )
