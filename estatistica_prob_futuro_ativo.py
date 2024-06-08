import pandas as pd
import csv
import probabilidade_preco_futuro_ativo
import carteira_aleatoria_otima_orientado_objeto

lista_ativos = ['ABEV3.SA', 'AZUL4.SA', 'B3SA3.SA', 'BBAS3.SA', 'BBDC3.SA', 'BBDC4.SA', 'BBSE3.SA', 'BRAP4.SA',
                'BRFS3.SA', 'BRKM5.SA', 'CASH3.SA', 'CCRO3.SA', 'CIEL3.SA', 'CMIG4.SA', 'COGN3.SA', 'CPFE3.SA',
                'CPLE6.SA', 'CRFB3.SA', 'CSAN3.SA', 'CSNA3.SA', 'CVCB3.SA', 'CYRE3.SA', 'DIRR3.SA', 'ECOR3.SA',
                'ELET3.SA', 'ELET6.SA', 'EMBR3.SA', 'ENGI11.SA', 'EQTL3.SA', 'EZTC3.SA', 'GGBR4.SA', 'GOAU4.SA',
                'GOLL4.SA', 'GRND3.SA', 'GUAR3.SA', 'HYPE3.SA', 'IRBR3.SA', 'ITSA4.SA', 'ITUB4.SA', 'JBSS3.SA',
                'KLBN11.SA', 'LREN3.SA', 'MGLU3.SA', 'MRFG3.SA', 'MRVE3.SA', 'MULT3.SA', 'NEOE3.SA', 'NTCO3.SA',
                'PCAR3.SA', 'PETR3.SA', 'PETR4.SA', 'POMO4.SA', 'POSI3.SA', 'PRIO3.SA', 'QUAL3.SA', 'RADL3.SA',
                'RAIL3.SA', 'RAPT4.SA', 'RENT3.SA', 'SANB11.SA', 'SBSP3.SA', 'SCAR3.SA', 'SUZB3.SA', 'TAEE11.SA',
                'TCSA3.SA', 'TECN3.SA', 'TIMS3.SA', 'TOTS3.SA', 'UGPA3.SA', 'USIM5.SA', 'VALE3.SA',
                'VIVT3.SA', 'WEGE3.SA', 'YDUQ3.SA']

inicio_dados = '2024-01-01'
final_dados = '2024-05-01'
valor_desejado = 1.12
otimizador = carteira_aleatoria_otima_orientado_objeto.PortfolioOptimization(lista_ativos, inicio_dados, final_dados)
ativo = otimizador.otimizar_carteira(valor_desejado)

# Lista para armazenar os resultados
resultados = []

# Executando a função 100 vezes e armazenando os resultados
for _ in range(100):
    resultado = probabilidade_preco_futuro_ativo.main(ativo)
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
print(ativo)



