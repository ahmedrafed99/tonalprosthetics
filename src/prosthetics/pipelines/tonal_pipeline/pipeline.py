from kedro.pipeline import Pipeline, node
import pandas as pd

from sklearn.preprocessing import StandardScaler

def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    # Supprimer les lignes avec des valeurs manquantes
    data.dropna(inplace=True)
    
    # Normaliser les colonnes numériques
    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    # Encoder les données catégoriques si nécessaire
    # data = pd.get_dummies(data)
    
    return data


def train_model(data: pd.DataFrame, model: tf.keras.Model, epochs=10):
    # Prétraitement des données si nécessaire
    # data = preprocess_data(data)

    # Séparation des données en entrées (X) et sorties (y)
    X_train = data.drop(columns=[
        'before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz',
        'before_exam_1000_Hz', 'before_exam_2000_Hz', 'before_exam_4000_Hz', 'before_exam_8000_Hz',
        'after_exam_125_Hz', 'after_exam_250_Hz', 'after_exam_500_Hz',
        'after_exam_1000_Hz', 'after_exam_2000_Hz', 'after_exam_4000_Hz', 'after_exam_8000_Hz'
    ])

    y_train = data[[
        'before_exam_125_Hz', 'before_exam_250_Hz', 'before_exam_500_Hz',
        'before_exam_1000_Hz', 'before_exam_2000_Hz', 'before_exam_4000_Hz', 'before_exam_8000_Hz',
        'after_exam_125_Hz', 'after_exam_250_Hz', 'after_exam_500_Hz',
        'after_exam_1000_Hz', 'after_exam_2000_Hz', 'after_exam_4000_Hz', 'after_exam_8000_Hz'
    ]]

    # Entraînement du modèle
    model.fit(X_train, y_train, epochs=epochs)

    return model


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
                func=train_model,
                inputs="preprocessed_data",
                outputs=None,
                name="train_model_node"
            ),
        ]
    )
