# Previsão e Classificação de Preços de Habitação (Housing)

Este repositório contém um projeto prático de *Machine Learning* focado na previsão e análise de preços de propriedades. O projeto foi desenvolvido no âmbito da unidade curricular de **TEI (Tópicos Especiais de Informática)** (3º Ano, 2º Semestre).

## 📌 Descrição do Projeto

O objetivo principal deste projeto é processar um conjunto de dados habitacional (`Housing.csv`) e treinar um modelo de inteligência artificial para prever os preços das casas com base em diversas características (ex: área, número de quartos, casas de banho, presença de ar condicionado, etc.). 

Para além da **Regressão Linear** (que tenta prever o preço exato), o projeto converte os resultados para um problema de **Classificação Binária**. Com base na média de preços, as casas são divididas em:
- **0 (Barata):** Casas cujo valor se encontra abaixo ou igual à média.
- **1 (Cara):** Casas cujo valor se encontra acima da média.

Esta divisão permite aplicar uma **Matriz de Confusão** para avaliar a capacidade que o sistema tem de identificar corretamente que casas pertencem à categoria de propriedades de alto valor e de baixo valor.

## 📁 Estrutura do Repositório

```text
tei_tp2/
│
├── data/
│   └── Housing.csv        # Conjunto de dados utilizado para o treino do modelo
│
├── src/
│   └── main.py            # Script principal com todo o código do projeto
│
├── Figure_1.png           # Gráficos e visualizações geradas pelo modelo
├── Figure_2.png
├── Figure_3.png
└── README.md              # Documentação do projeto
```

## 🛠️ Tecnologias Utilizadas

- **Python** (Linguagem de programação principal)
- **Pandas** (Tratamento, leitura e análise de dados)
- **Scikit-Learn** (Criação do modelo de Regressão Linear e métricas de avaliação)
- **Matplotlib** (Criação de gráficos e visualização de resultados)

## 🚀 Como Executar o Projeto

1. Certifique-se de que tem o Python instalado no seu computador.
2. (Opcional) Crie um ambiente virtual para o projeto.
3. Instale as dependências necessárias através da linha de comandos (terminal):
   ```bash
   pip install pandas scikit-learn matplotlib seaborn
   ```
4. Navegue até à pasta do projeto e execute o ficheiro principal:
   ```bash
   python src/main.py
   ```

## 📊 Métricas e Avaliação

O script irá apresentar diversas métricas no terminal:
- **R2 Score & MAE:** Para avaliar a margem de erro contínua da regressão linear.
- **Relatório de Performance (Classification Report):**
  - **Precision (Precisão):** Avalia a percentagem de acertos quando o modelo declara que uma casa é "Cara" ou "Barata".
  - **Recall (Sensibilidade):** Avalia a percentagem de casas reais de uma categoria que o modelo conseguiu detetar corretamente.
  - **F1-Score:** Uma média harmónica entre as métricas anteriores para avaliar a consistência geral do modelo.

Ao terminar, o script também exibe gráficos de regressão (`Real vs Previsto`) para fácil interpretação visual do erro, bem como a Matriz de Confusão.
