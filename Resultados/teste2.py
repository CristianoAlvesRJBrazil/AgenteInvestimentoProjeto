import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
import pandas as pd

# Obter dados históricos
ticker = 'EMBR3.SA'
data = yf.download(ticker, start='2024-01-01', end='2024-05-02')
precos = data['Close']

# Calcular retornos diários
retornos = precos.pct_change().dropna()

# Ajustar o modelo GARCH(1,1)
modelo = arch_model(retornos, vol='Garch', p=1, q=1)
modelo_ajustado = modelo.fit(disp='off')

# Prever a volatilidade futura (60 dias à frente)
horizonte = 30
previsao = modelo_ajustado.forecast(horizon=horizonte)
volatilidade_futura = np.sqrt(previsao.variance.values[-1,:])

# Calcular a tendência histórica
tendencia_historica = retornos.mean()  # Média dos retornos diários históricos

# Simulação de Monte Carlo com tendência
n_simulacoes = 1000  # Número de trajetórias simuladas
preco_atual = precos.iloc[-1]
simulacoes = np.zeros((horizonte, n_simulacoes))

np.random.seed(42)  # Para reprodutibilidade

for i in range(n_simulacoes):
    retornos_simulados = np.random.normal(tendencia_historica, volatilidade_futura)
    simulacoes[:, i] = preco_atual * np.exp(np.cumsum(retornos_simulados))

# Criar um índice de datas futuro
ultima_data = precos.index[-1]
datas_futuras = pd.date_range(start=ultima_data, periods=horizonte + 1, freq='B')[1:]

# Calcular a média dos preços simulados
precos_futuros_medio = simulacoes.mean(axis=1)

# Calcular a probabilidade de que o preço futuro esteja acima do preço atual
precos_futuros_finais = simulacoes[-1, :]
probabilidade_acima = np.mean(precos_futuros_finais > preco_atual)

# Exibir a probabilidade
print(f'Probabilidade do preço estar acima de {preco_atual:.2f} em {horizonte} dias: {probabilidade_acima:.2%}')

# Validar com Histograma
plt.figure(figsize=(10, 6))
plt.hist(precos_futuros_finais, bins=50, alpha=0.7, color='blue', label='Preços Futuros Finais')
plt.axvline(preco_atual, color='red', linestyle='--', label=f'Último Preço ({preco_atual:.2f})')
plt.title(f'Histograma dos Preços Finais em {horizonte} Dias')
plt.xlabel('Preço Final')
plt.ylabel('Frequência')
plt.legend()
plt.grid(True)
plt.show()

# Plotar os preços futuros simulados
plt.figure(figsize=(10, 6))
for i in range(n_simulacoes):
    plt.plot(datas_futuras, simulacoes[:, i], color='blue', alpha=0.1)  # Trajetórias simuladas
plt.plot(datas_futuras, precos_futuros_medio, color='red', label='Preço Médio Simulado', linewidth=2)
plt.axhline(preco_atual, color='green', linestyle='--', label=f'Último Preço ({preco_atual:.2f})')
plt.title(f'Simulações de Preços Futuros para {ticker}')
plt.xlabel('Data')
plt.ylabel('Preço')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Comparar com tendência histórica
plt.figure(figsize=(10, 6))
plt.plot(precos.index, precos, label='Preços Históricos', color='black')
plt.axhline(preco_atual, color='green', linestyle='--', label=f'Último Preço ({preco_atual:.2f})')
plt.plot(datas_futuras, precos_futuros_medio, color='red', label='Preço Médio Simulado', linewidth=2)
plt.title(f'Comparação de Preços Históricos e Simulados para {ticker}')
plt.xlabel('Data')
plt.ylabel('Preço')
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
