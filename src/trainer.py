import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import Settings
from model_factory import ModelFactory


class Trainer:
    def __init__(self):
        self.settings = Settings()

    def train(self, data_path: str):
        # Carrega o dataset único
        df = pd.read_csv(data_path)

        # Separação de features e target
        X = df.drop(self.settings.TARGET_COLUMN, axis=1)
        y = df[self.settings.TARGET_COLUMN]

        # Split treino / teste
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=self.settings.RANDOM_STATE,
            stratify=y
        )

        # Pipeline de pré-processamento + modelo
        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", ModelFactory.create())
        ])

        # Configuração do experimento no MLflow
        mlflow.set_experiment(self.settings.EXPERIMENT_NAME)

        with mlflow.start_run():
            # Treinamento
            pipeline.fit(X_train, y_train)

            # Avaliação
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            # Log de métricas
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)

            # Registro do modelo
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="model",
                registered_model_name="WaterPotabilityModel"
            )

            print(f"Accuracy: {acc:.4f} | F1-score: {f1:.4f}")
