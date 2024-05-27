"""
This is a boilerplate pipeline 'tonal_pipeline'
generated using Kedro 0.19.5
"""
from typing import Dict, Any
import pandas as pd

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    data.dropna(inplace=True)
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    return data
def split_data(data: pd.DataFrame, test_size: float, random_state: int):
    X = data.drop(columns=[
        'before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz',
        'before_exam_1000_Hz', 'before_exam_2000_Hz', 'before_exam_4000_Hz', 'before_exam_8000_Hz',
        'after_exam_125_Hz', 'after_exam_250_Hz', 'after_exam_500_Hz',
        'after_exam_1000_Hz', 'after_exam_2000_Hz', 'after_exam_4000_Hz', 'after_exam_8000_Hz'
    ])
    y = data[[
        'before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz',
        'before_exam_1000_Hz', 'before_exam_2000_Hz', 'before_exam_4000_Hz', 'before_exam_8000_Hz',
        'after_exam_125_Hz', 'after_exam_250_Hz', 'after_exam_500_Hz',
        'after_exam_1000_Hz', 'after_exam_2000_Hz', 'after_exam_4000_Hz', 'after_exam_8000_Hz'
    ]]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def create_model(units, activation, l2_value, dropout_rate, learning_rate):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units, activation=activation, kernel_regularizer=tf.keras.regularizers.l2(l2_value)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(14, activation='linear')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='mean_squared_error',
                  metrics=['mse'])
    return model

def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, model: tf.keras.Model, epochs=10):
    model.fit(X_train, y_train, epochs=epochs)
    from tensorflow.keras import layers, models, regularizers
    
    input_shape = (data.shape[1], 1)
    units = parameters.get("units", 128)
    activation = parameters.get("activation", "relu")
    l2_value = parameters.get("l2_value", 0.01)
    dropout_rate = parameters.get("dropout_rate", None)
    learning_rate = parameters.get("learning_rate", 0.001)
    
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(32, 3, activation=activation)(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.ZeroPadding1D(1)(x)
    x = layers.Conv1D(64, 3, activation=activation)(x)
    x = layers.ZeroPadding1D(1)(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(units, activation='relu', kernel_regularizer=regularizers.l2(l2_value))(x)
    
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)
    
    outputs = layers.Dense(input_shape[0], activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    model.fit(data, data, epochs=10)  # Supposons que les données sont correctement formatées
    
    return model

def evaluate_model(model: Any, test_data: pd.DataFrame) -> Dict[str, Any]:
    # Évaluation du modèle
    evaluation = model.evaluate(test_data, test_data)
    return {"loss": evaluation[0], "accuracy": evaluation[1]}
