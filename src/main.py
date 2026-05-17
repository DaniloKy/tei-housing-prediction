import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score, confusion_matrix, classification_report

# ==========================================
# Seleção: Descrição do dataset e variáveis
# ==========================================
print("--- Seleção ---")
# Descrição do conjunto de dados (dataset) Housing.csv e escolha das 7 variáveis principais.
df = pd.read_csv('../data/Housing.csv')

# Escolha das 7 variáveis principais para X e o alvo y
features = ['area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'airconditioning']
X = df[features].copy()
y = df['price'].copy()

# ==========================================
# Pré-processamento: Limpeza e nulos
# ==========================================
print("\n--- Pré-processamento ---")
# Limpeza de dados e tratamento de valores nulos
print(f"Valores nulos antes do tratamento:\n{X.isnull().sum()}")

# Tratamento de valores nulos (preenchimento com mediana para numéricos e moda para categóricos)
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].fillna(X[col].mode()[0])
    else:
        X[col] = X[col].fillna(X[col].median())
y = y.fillna(y.median())
print("Tratamento de nulos concluído.")

# ==========================================
# Transformação: Normalização e Binarização
# ==========================================
print("\n--- Transformação ---")
# A normalização (StandardScaler) e a binarização das variáveis (Sim/Não para 0/1).
binary_columns = ['mainroad', 'guestroom', 'airconditioning']
for col in binary_columns:
    X[col] = X[col].map({'yes': 1, 'no': 0})

# Divisão em Treino e Teste antes de normalizar
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalização (StandardScaler)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("Binarização e Normalização (StandardScaler) concluídas.")

# ==========================================
# Data Mining: Regressão Linear
# ==========================================
print("\n--- Data Mining ---")
# Substituição do algoritmo "Random Forest" pela Regressão Linear
modelo = LinearRegression()
modelo.fit(X_train_scaled, y_train)
print("Modelo de Regressão Linear treinado com sucesso (substituindo Random Forest).")

# ==========================================
# Identificação das Variáveis Mais Importantes (Feature Importance)
# ==========================================
print("\n--- Feature Importance ---")
importancias = pd.DataFrame({
    'Variável': features,
    'Importância (Coeficiente)': modelo.coef_
})
importancias['Importância Absoluta'] = importancias['Importância (Coeficiente)'].abs()
importancias = importancias.sort_values(by='Importância Absoluta', ascending=False)
print(importancias[['Variável', 'Importância (Coeficiente)']])

plt.figure(figsize=(8, 5))
sns.barplot(x='Importância (Coeficiente)', y='Variável', data=importancias, palette='viridis', hue='Variável', legend=False)
plt.title('Feature Importance (Coeficientes da Regressão Linear)')
plt.show()

# ==========================================
# Avaliação: Matriz e Métricas
# ==========================================
print("\n--- Avaliação ---")
# Uso da Matriz de Confusão e métricas de precisão/recall
y_pred = modelo.predict(X_test_scaled)

mae = mean_absolute_error(y_test, y_pred)
print(f"Erro Médio Absoluto (MAE): {mae:.2f}")
print(f"R2 Score: {r2_score(y_test, y_pred):.2f}")

# Visualização da Regressão
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Preço Real')
plt.ylabel('Previsão do Sistema')
plt.title('Regressão Linear: Real vs Previsto')
plt.show()

# Uso da Matriz de Confusão e métricas de precisão/recall
limiar = y.mean()
y_test_cat = (y_test > limiar).astype(int)
y_pred_cat = (y_pred > limiar).astype(int)

print("\nRelatório de Performance (Precisão e Recall):")
print(classification_report(y_test_cat, y_pred_cat, target_names=['Barata', 'Cara']))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test_cat, y_pred_cat), annot=True, fmt='d', cmap='Greens', xticklabels=['Barata', 'Cara'], yticklabels=['Barata', 'Cara'])
plt.title('Matriz de Confusão')
plt.ylabel('Realidade', fontweight='bold')
plt.xlabel('Previsão', fontweight='bold')
plt.show()

# 2. Testar diferentes divisões (Split Ratios)
splits = [0.3, 0.2, 0.1]

print("\nComparação de Performance por Divisão de Dados:")
print("-" * 45)

for test_size in splits:
    X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Normalização
    X_train_split_scaled = scaler.fit_transform(X_train_split)
    X_test_split_scaled = scaler.transform(X_test_split)
    
    # Treino
    modelo_split = LinearRegression()
    modelo_split.fit(X_train_split_scaled, y_train_split)
    
    # Avaliação
    y_pred_split = modelo_split.predict(X_test_split_scaled)
    score = r2_score(y_test_split, y_pred_split)
    
    treino_perc = int((1 - test_size) * 100)
    teste_perc = int(test_size * 100)
    print(f"Treino: {treino_perc}% | Teste: {teste_perc}% -> R2 Score: {score:.4f}")