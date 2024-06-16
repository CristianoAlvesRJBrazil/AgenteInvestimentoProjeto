# Importação das bibliotecas necessárias
import numpy as np
import pandas as pd
import yfinance as yf
import probabilidade_futuro_sem_grafico as pfsg
import estatistica_prob_futuro_carteira_v2 as epfc2

def preco_atual_real(ticker, inicio_dados, final_dados):
    
    # Obter dados históricos do estoque
    dados_estoque_atual_real = yf.download(ticker, start=inicio_dados, end=final_dados)['Close'].mean(axis=1)

    dados_estoque_atual_real = dados_estoque_atual_real[-1] 

    #print(dados_estoque_atual_real[-1])
    return dados_estoque_atual_real


def preco_futuro_real(ticker, inicio_dados, final_dados):
    
    # Obter dados históricos do estoque
    dados_estoque_futuro_real = yf.download(ticker, start=inicio_dados, end=final_dados)['Close'].mean(axis=1)

    dados_estoque_futuro_real = dados_estoque_futuro_real.max()
    
    #print(dados_estoque_futuro_real.max())    
    return dados_estoque_futuro_real

    
if __name__ == "__main__":
    # Buscando dados da carteira - Intervalo Inicial Real
    ticker = ['CPFE3.SA', 'ECOR3.SA', 'EMBR3.SA', 'SMFT3.SA']
    inicio_dados = '2023-01-01'
    final_dados = '2023-05-01'
    preco_atual_real = preco_atual_real(ticker, inicio_dados, final_dados)
    
    # Simulando a probabilidade do preço futuro está acima do preço atual - Monte Carlo
    num_simulacoes = 1000
    num_dias = 60    
    probabilidade_acima_preco_atual, preco_futuro_simulado = pfsg.main(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)
    
    # Buscando dados da carteira - Intervalo Futuro Real
    inicio_dados = '2023-05-01'
    final_dados = '2023-08-01'
    preco_futuro_real = preco_futuro_real(ticker, inicio_dados, final_dados)
    
    # Calculando os Retornos Acumulados do Real e do Simulado    
    Retorno_Acumulado_Real = preco_futuro_real / preco_atual_real    
    Retorno_Acumulado_Simulado = preco_futuro_simulado / preco_atual_real
    
    # Resultado do Backtesting
    dados = epfc2.ler_dados_csv()
    print(f"Probabilidade mediana do preço futuro estar acima do esperado: {round(dados['Probabilidade'].median(), 4) * 100}%")
    print(f"Probabilidade do preço futuro estar acima do preço atual simulado agora: {probabilidade_acima_preco_atual:.2%}")        
    print(f"Retono_Acumulado_Simulado = {Retorno_Acumulado_Simulado}")
    print(f"Retono_Acumulado_Real = {Retorno_Acumulado_Real}")
    
    
    
    