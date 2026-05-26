# 🔍 anomaly-detection-credit-card-transactions

Estudo prático sobre **detecção de anomalias em transações financeiras** com Machine Learning — do pré-processamento à explicabilidade de modelos avançados.

---

## 📌 Contexto

Fraudes em cartões de crédito representam um dos maiores desafios para instituições financeiras ao redor do mundo. A identificação de transações fraudulentas em tempo real exige modelos capazes de aprender padrões sutis em meio a um volume massivo de dados legítimos.

Este projeto utiliza o dataset público de transações de cartão de crédito disponibilizado pelo TensorFlow/Google, que contém registros reais anonimizados por meio de **PCA (Análise de Componentes Principais)** para preservar a privacidade dos titulares. O conjunto de dados apresenta um forte **desbalanceamento de classes** — característica central do problema e ponto de atenção em todo o desenvolvimento.

> **Fonte do dataset:**
> `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv`

---

## 🎯 Objetivos de Estudo

Este material foi organizado em três etapas progressivas, cada uma aprofundando um aspecto diferente do pipeline de Machine Learning para detecção de fraudes:

### 1️⃣ Introdução — Exploração e Pré-processamento dos Dados
- Compreender a estrutura do dataset (colunas `Time`, `V1`–`V28`, `Amount`, `Class`)
- Identificar e quantificar o desbalanceamento entre classes (fraude × não fraude)
- Aplicar **Feature Engineering**: transformação logarítmica (`Amount_log`) e padronização via `StandardScaler`
- Preparar os dados para treinamento com `train_test_split` (estratificado)

### 2️⃣ Avaliação do Modelo e Técnicas de Balanceamento
- Treinar um modelo base de **Regressão Logística** e interpretar suas métricas
- Entender por que a **Acurácia isolada é enganosa** em datasets desbalanceados
- Analisar as curvas **ROC** e **Precision-Recall** como métricas mais adequadas ao problema
- Aplicar técnicas de balanceamento:
  - **Undersampling**: redução da classe majoritária
  - **Oversampling (SMOTE)**: criação de exemplos sintéticos da classe minoritária

### 3️⃣ Modelos Avançados e Explicabilidade
- Treinar e avaliar um modelo de **Random Forest** com controle de profundidade e balanceamento de pesos
- Organizar o fluxo de treinamento com **Pipeline** (`StandardScaler` + `LogisticRegression`)
- Ajustar o **Threshold de decisão** para priorizar Recall (identificar mais fraudes)
- Aplicar o modelo **XGBoost** com ajuste de peso (`scale_pos_weight`) para lidar com o desbalanceamento
- Interpretar a **Importância das Variáveis** (feature importances) gerada pelo XGBoost
- Utilizar **GridSearchCV** para otimização automática de hiperparâmetros

---

## 🗂️ Estrutura do Repositório

```
anomaly-detection-credit-card-transactions/
│
├── notebook/
│   └── anomaly_detection.ipynb         # Notebook completo com todo o código do projeto
│
├── docs/
│   ├── 1_introducao.docx               # Introdução e pré-processamento
│   ├── 2_avaliacao_balanceamento.docx  # Avaliação e técnicas de balanceamento
│   └── 3_modelos_avancados.docx        # Modelos avançados e explicabilidade
│
└── README.md
```

---

## 🚀 Como Executar

O notebook pode ser executado diretamente no navegador, sem instalação local:

[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VGi01InlAyXnrS0ziuN0x2G6DKGEbXUf#scrollTo=EpGB_-rX1Ke0)

---

## 🛠️ Tecnologias e Bibliotecas

| Biblioteca | Finalidade |
|---|---|
| `pandas` | Manipulação e análise dos dados |
| `numpy` | Operações numéricas e transformações |
| `scikit-learn` | Pré-processamento, modelos e avaliação |
| `imbalanced-learn` | Técnicas de balanceamento (SMOTE) |
| `xgboost` | Modelo avançado de gradient boosting |
| `matplotlib` | Visualização de curvas e gráficos |

---

## 📊 Métricas Utilizadas

Dado o forte desbalanceamento do dataset, as métricas principais utilizadas são:

- **Precision**: proporção de transações classificadas como fraude que realmente são fraudes
- **Recall**: proporção de fraudes reais que o modelo conseguiu identificar
- **F1-Score**: equilíbrio entre Precision e Recall
- **Curva ROC / AUC**: desempenho do modelo em diferentes limiares de decisão
- **Curva Precision-Recall**: comportamento do modelo no trade-off entre Precision e Recall

---

## 📝 Observações

- As colunas `V1`–`V28` são componentes resultantes de PCA aplicado aos dados originais, garantindo a anonimização das informações dos clientes.
- A coluna `Class` é o alvo do modelo: `0` = transação legítima, `1` = transação fraudulenta.
- O desbalanceamento de classes é a principal dificuldade do problema e motivou o estudo de diferentes estratégias de balanceamento e escolha de métricas.
