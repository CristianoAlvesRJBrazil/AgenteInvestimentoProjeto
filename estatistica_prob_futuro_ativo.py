import pandas as pd
import csv
import probabilidade_preco_futuro_ativo

# Lista para armazenar os resultados
resultados = []

# Executando a função 100 vezes e armazenando os resultados
for _ in range(100):
    resultado = probabilidade_preco_futuro_ativo.main()
    resultados.append(resultado)

# Escrevendo os resultados em um arquivo CSV
with open('resultados.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(['Resultado'])  # Escreve o cabeçalho
    for resultado in resultados:
        writer.writerow([resultado])

# Ler um arquivo CSV e armazenar os dados em um DataFrame
df = pd.read_csv('resultados.csv')

# Criar resumo estatistico de df
print(f"Resumo estatístico:\n {df.describe()}.")

# Calcular a media do df
print(f"A media é: {round(df['Resultado'].mean(), 4)}.")

# Calcular a mediana do df
print(f"A mediada é: {round(df['Resultado'].median(), 4)}.")

# Calcular a moda do df
print(f"A moda é: {round(df['Resultado'].mode(), 4)}.")

print(f"Probabilidade do preço futuro estar acima do esperado: {round(df['Resultado'].median(), 4)*100}%")
print()



