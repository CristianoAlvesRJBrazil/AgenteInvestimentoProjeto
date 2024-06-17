import pandas as pd
import yfinance as yf

df = pd.read_csv('Resultados/resultados08.csv')
title_column = df.columns[0]
ticker = [title_column[2:10], title_column[14:22], title_column[26:34], title_column[38:46]]
ticker = ticker[0]

# Buscando dados da carteira - Intervalo Futuro Real
inicio_dados = '2023-05-01'
final_dados = '2023-08-01'

dados_estoque_futuro_real_maximo = yf.download(ticker, start=inicio_dados, end=final_dados)['High']
dados_estoque_futuro_real_maximo.max()

dados_estoque_futuro_real_close_futuro = yf.download(ticker, start=inicio_dados, end=final_dados)['Close']
dados_estoque_futuro_real_close_futuro.max()

inicio_dados = '2023-04-25'
final_dados = '2023-05-01'
dados_estoque_futuro_real_close = yf.download(ticker, start=inicio_dados, end=final_dados)['Close']
dados_estoque_futuro_real_close

print(dados_estoque_futuro_real_close[-1])
print(dados_estoque_futuro_real_maximo.max())
print(dados_estoque_futuro_real_close_futuro.max())







