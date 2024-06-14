import pandas as pd
import csv
import carteira_aleatoria_otima_orientado_objeto
import lista_ativos_setores
import MonteCarloSimulationCarteiraOB

# Lista dos ativos da B3
lista_ativos = lista_ativos_setores.lista_atual_ibxx_02
# Selecionar a carteira vencedora
inicio_dados = '2024-01-01'
final_dados = '2024-05-01'
valor_desejado = 1.24
otimizador = carteira_aleatoria_otima_orientado_objeto.PortfolioOptimization(lista_ativos, inicio_dados, final_dados)
carteira_vencedora = otimizador.otimizar_carteira(valor_desejado)
print(f"Carteira_Vencedora = {carteira_vencedora}.")

# Realizar simulação de Monte Carlo
ticker = carteira_vencedora
num_simulacoes = 1000
num_dias = 60
simulacao = MonteCarloSimulationCarteiraOB.MonteCarloSimulation(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)

# Lista para armazenar os resultados
resultados = []

# Executando a função 100 vezes e armazenando os resultados
for _ in range(100):
    precos_simulados_df = simulacao.simular_precos()
    probabilidade_acima_preco_atual = simulacao.calcular_probabilidade_acima_preco_atual()
    resultados.append(probabilidade_acima_preco_atual)

# Escrevendo os resultados em um arquivo CSV
with open('resultados.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow([ticker])  # Escreve o cabeçalho
    for resultado in resultados:
        writer.writerow([resultado])

# Ler um arquivo CSV e armazenar os dados em um DataFrame
df = pd.read_csv('resultados.csv')

# Criar resumo estatistico de df
print(f"Resumo estatístico:\n {df.describe()}.")

# Calcular a media do df
print(f"A media é: {round(df.mean(), 4)}.")

# Calcular a mediana do df
print(f"A mediada é: {round(df.median(), 4)}.")

# Calcular a moda do df
print(f"A moda é: {round(df.mode(), 4)}.")

print(f"Probabilidade do preço futuro estar acima do esperado: {round(df.median(), 4)*100}%")
print(f"Carteira_Vencedora = {carteira_vencedora}.")



