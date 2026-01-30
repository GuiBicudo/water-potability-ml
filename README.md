\# Water Potability Classification



Este projeto implementa um modelo de Machine Learning para classificar se a água é potável ou não, utilizando Scikit-learn e MLflow para rastreamento e versionamento do modelo.



\## Dataset

O dataset utilizado está disponível no Kaggle:

https://www.kaggle.com/datasets/adityakadiwal/water-potability



O arquivo `water\_potability.csv` contém atributos físico-químicos da água e uma variável alvo indicando a potabilidade.



\## Tecnologias Utilizadas

\- Python 3.10

\- Pandas

\- Scikit-learn

\- MLflow

\- Pydantic



\## Estrutura do Projeto



water-potability-ml/

├── data/

│ └── water\_potability.csv

├── mlruns/

│ └── artefatos e métricas do MLflow

├── src/

│ ├── config.py

│ ├── main.py

│ ├── model\_factory.py

│ ├── schemas.py

│ └── trainer.py

├── README.md

├── requirements.txt

└── .gitignore





\## Execução do Projeto



\### 1. Instalação das dependências

```bash

python -m pip install -r requirements.txt

```



\### 2. Treinamento e Versionamento do modelo

```bash

python src/main.py

```



Durante a execução:



O modelo é treinado



Métricas de avaliação são calculadas



O modelo é registrado no MLflow Model Registry



\## 3. Visualização dos experimentos



```bash
mlflow ui
```



Acesse no navegador:

http://localhost:5000



Resultados



As execuções do treinamento são registradas no MLflow, contendo métricas como Accuracy e F1-score, além do versionamento do modelo treinado.







