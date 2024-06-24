import numpy as np
import pandas as pd
import yfinance as yf
import probabilidade_futuro_sem_grafico as pfsg
import estatistica_prob_futuro_carteira_v2 as epfcv2

class PrecoEstoque:
    def __init__(self):
        pass
    
    def obter_preco_atual(self, ticker, inicio_dados, final_dados):
        dados_estoque_atual_real = yf.download(ticker, start=inicio_dados, end=final_dados)['Close'].mean(axis=1)
        return dados_estoque_atual_real[-1]
    
    def obter_preco_futuro(self, ticker, inicio_dados, final_dados):
        dados_estoque_futuro_real = yf.download(ticker, start=inicio_dados, end=final_dados)['Close'].mean(axis=1)
        return dados_estoque_futuro_real.max()
    
    def calcular_retorno_acumulado(self, preco_futuro, preco_atual):
        return preco_futuro / preco_atual

if __name__ == "__main__":
    # Buscando a carteira para o Backtesting, inserindo os intervalos para analise e outros
    df = pd.read_csv('resultados.csv')
    title_column = df.columns[0]
    ticker = [title_column[2:10], title_column[14:22], title_column[26:34], title_column[38:46]]
    inicio_dados = '2023-01-01'
    final_dados = '2023-05-01'
    valor_desejado = 1.20
    num_simulacoes = 100
    num_dias = 60
    
    # Executando o método executar_simulações do código que cálcula a probabilidade do preço futuro da carteira
    resultados = epfcv2.executar_simulacoes(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)
    epfcv2.salvar_resultados_csv(resultados) 
    
    # Obtendo o Preço Atual do ativo para um intervalo desejado ao instanciar a classe PrecoEstoque       
    preco_estoque = PrecoEstoque()
    inicio_dados = '2023-01-01'
    final_dados = '2023-05-01'
    preco_atual_real = preco_estoque.obter_preco_atual(ticker, inicio_dados, final_dados)
    
    # Simulando a probabilidade do preço futuro estar acima do preço atual - Monte Carlo
    probabilidade_acima_preco_atual, preco_futuro_simulado = pfsg.main(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)
    
    # Buscando dados da carteira - Intervalo Futuro Real
    inicio_dados = '2023-05-01'
    final_dados = '2023-08-01'
    preco_futuro_real = preco_estoque.obter_preco_futuro(ticker, inicio_dados, final_dados)
    
    # Calculando os Retornos Acumulados do Real e do Simulado    
    retorno_acumulado_real = preco_estoque.calcular_retorno_acumulado(preco_futuro_real, preco_atual_real)    
    retorno_acumulado_simulado = preco_estoque.calcular_retorno_acumulado(preco_futuro_simulado, preco_atual_real)
    
    # Resultado do Backtesting
    dados = epfcv2.ler_dados_csv()
    print()
    print(f"Resultado do Backtesting para carteira: {ticker}")
    print(f"Probabilidade mediana do preço futuro estar acima do esperado: {round(dados['Probabilidade'].median(), 4) * 100}%")
    print(f"Probabilidade do preço futuro estar acima do preço atual simulado agora: {probabilidade_acima_preco_atual:.2%}")        
    print(f"Retorno_Acumulado_Simulado = {retorno_acumulado_simulado}")
    print(f"Retorno_Acumulado_Real = {retorno_acumulado_real}")
    
    if retorno_acumulado_simulado and retorno_acumulado_real > 1.00:
        print(f"O Agente acertou a tendência de alta!!!")
    else:
        print(f"O Agente não acertou a tendência!!!")

      
    #Carteira_Vencedora = ['TEND3.SA', 'MOVI3.SA', 'RRRP3.SA', 'ECOR3.SA'].