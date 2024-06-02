# prompt: buscar os ativos que possui as maiores correlações com o ativo selecionado.

import numpy as np
import pandas as pd
import yfinance as yf
#import matplotlib.pyplot as plt

# Selecione o ativo desejado
ativo_selecionado = 'MGLU3.SA'

# Obtenha os dados de preços dos ativos
ativos = ['BRAP4.SA', 'CPFE3.SA', 'VALE3.SA', 'PETR4.SA', 'CIEL3.SA', 'MGLU3.SA', 'ELET3.SA', 'IRBR3.SA', 'ABEV3.SA',
          'CSNA3.SA', 'BRPR3.SA', 'ITSA4.SA', 'KLBN11.SA', 'GGBR4.SA', 'BBDC4.SA', 'LREN3.SA', 'CPLE6.SA', 'UGPA3.SA',
          'NTCO3.SA', 'EMBR3.SA', 'RENT3.SA', 'CYRE3.SA', 'EQTL3.SA', 'ITUB4.SA', 'SUZB3.SA', 'BRFS3.SA', 'CMIG4.SA',
          'B3SA3.SA', 'WEGE3.SA', 'BBAS3.SA', 'ECOR3.SA', 'BBDC3.SA', 'CCRO3.SA', 'BBSE3.SA', 'CSAN3.SA', 'BRKM5.SA']
dados = yf.download(ativos, period='1y')['Close']

# Calcule a matriz de correlação
matriz_correlacao = dados.corr()

# Obtenha as correlações com o ativo selecionado
correlacoes = matriz_correlacao[ativo_selecionado]

# Classifique os ativos em ordem decrescente de correlação
ativos_ordenados = correlacoes.sort_values(ascending=False)

# Imprima os ativos com as maiores correlações
print("Ativos com as maiores correlações com", ativo_selecionado)
print(ativos_ordenados.head(10))