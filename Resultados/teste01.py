import numpy as np
import pandas as pd
import yfinance as yf


class PrecoEstoque:
    def __init__(self):
        pass
    
    def obter_preco_atual(self, ticker, inicio_dados, final_dados):
        """A função retorna o preço atual da carteira"""
        dados_estoque_atual_real = yf.download(ticker, start=inicio_dados, end=final_dados)['Close'].mean(axis=1)
        return dados_estoque_atual_real[-1]
    
    def obter_preco_futuro(self, ticker, inicio_dados, final_dados):
        """A função procura o maior preço no futuro que a carteira alcançou"""
        dados_estoque_futuro_real = yf.download(ticker, start=inicio_dados, end=final_dados)['Close'].mean(axis=1)
        return dados_estoque_futuro_real.max()
    
    def calcular_retorno_acumulado(self, preco_futuro, preco_atual):
        """A função calcula o retorno para o período"""
        return preco_futuro / preco_atual

if __name__ == "__main__":
    # Buscando a carteira para o Backtesting, inserindo os intervalos para analise e outros
    df = pd.read_csv('Resultados/resultados01.csv')
    title_column = df.columns[0]
    ticker = [title_column[2:10], title_column[14:22], title_column[26:34], title_column[38:46]]
    inicio_dados = '2023-01-01'
    final_dados = '2023-05-01'
    valor_desejado = 1.20
    num_simulacoes = 100
    num_dias = 60
    
    # Obtendo o Preço Atual do ativo para um intervalo desejado ao instanciar a classe PrecoEstoque       
    preco_estoque = PrecoEstoque()
    inicio_dados = '2023-01-01'
    final_dados = '2023-05-01'
    preco_atual_real = preco_estoque.obter_preco_atual(ticker, inicio_dados, final_dados)
    
    # Buscando dados da carteira - Intervalo Futuro Real
    inicio_dados = '2023-05-01'
    final_dados = '2023-08-01'
    preco_futuro_real = preco_estoque.obter_preco_futuro(ticker, inicio_dados, final_dados)
    
    # Calculando os Retornos Acumulados do Real e do Simulado    
    retorno_acumulado_real = preco_estoque.calcular_retorno_acumulado(preco_futuro_real, preco_atual_real)    
    retorno_acumulado_simulado = preco_estoque.calcular_retorno_acumulado(25.193741, preco_atual_real)
    
    # Resultado do Backtesting
    print()
    print(f"Resultado do Backtesting para carteira: {ticker}")
    print(f"Retorno_Acumulado_Simulado = {retorno_acumulado_simulado}")
    print(f"Retorno_Acumulado_Real = {retorno_acumulado_real}")
    
    if retorno_acumulado_simulado and retorno_acumulado_real > 1.00:
        print(f"O Agente acertou a tendência de alta!!!")
    else:
        print(f"O Agente não acertou a tendência!!!")

  # FIM
    
    
    
    
    
    
    
    