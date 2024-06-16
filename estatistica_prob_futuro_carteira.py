import pandas as pd
import csv
import matplotlib.pyplot as plt
import probabilidade_futuro_sem_grafico as pfsg

# Lista de tickers para simulação (não usada diretamente no código fornecido)
ticker = ['IRBR3.SA', 'TEND3.SA', 'NTCO3.SA', 'IGTI11.SA']
inicio_dados = '2023-01-01'
final_dados = '2023-05-01'
num_simulacoes = 100
num_dias = 60

# Lista para armazenar os resultados
resultados = []

# Executando a função 100 vezes e armazenando os resultados
for _ in range(100):
    probabilidade_acima_preco_atual, precos_simulados_max = pfsg.main(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)
    resultados.append(probabilidade_acima_preco_atual)

# Escrevendo os resultados em um arquivo CSV
with open('resultados.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Probabilidade'])  # Escreve o cabeçalho
    for resultado in resultados:
        writer.writerow([resultado])

# Ler o arquivo CSV e armazenar os dados em um DataFrame
df = pd.read_csv('resultados.csv')

# Criar resumo estatístico do DataFrame
print(f"Resumo estatístico:\n{df.describe()}")

# Calcular a média do DataFrame
print(f"Média: {round(df['Probabilidade'].mean(), 4)}")

# Calcular a mediana do DataFrame
print(f"Mediana: {round(df['Probabilidade'].median(), 4)}")

# Calcular a moda do DataFrame
print(f"Moda:\n{df['Probabilidade'].mode()}")

# Calcular a probabilidade do preço futuro estar acima do esperado
print(f"Probabilidade do preço futuro estar acima do esperado: {round(df['Probabilidade'].median(), 4) * 100}%")

# Gerar histograma dos resultados
plt.figure(figsize=(10, 6))
plt.hist(df['Probabilidade'], bins=20, color='skyblue', edgecolor='black')
plt.title('Histograma das Probabilidades de Preços Futuros')
plt.xlabel('Probabilidade')
plt.ylabel('Frequência')
plt.grid(True)
plt.show()

