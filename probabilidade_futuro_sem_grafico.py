import numpy as np
import pandas as pd
import yfinance as yf

def simulacao_monte_carlo(ticker, num_simulacoes, num_dias, inicio_dados, final_dados):
    """ Realiza simulação de Monte Carlo para uma carteria de investimento. """
    try:
        # Obter dados históricos do estoque
        dados_estoque = yf.download(ticker, start=inicio_dados, end=final_dados)['Close'].mean(axis=1)

        # Calcular retornos diários
        retornos_diarios = dados_estoque.pct_change()

        # Calcular média e desvio padrão dos retornos diários
        retorno_medio = retornos_diarios.mean()
        desvio_padrao = retornos_diarios.std()

        # Gerar números aleatórios para as simulações
        numeros_aleatorios = np.random.normal(size=(num_simulacoes, num_dias))

        # Inicializar preços simulados do estoque
        precos_simulados = np.zeros((num_simulacoes, num_dias))
        precos_simulados[:, 0] = dados_estoque.iloc[-1]

        # Simular preços do estoque para cada dia
        for i in range(1, num_dias):
            precos_simulados[:, i] = precos_simulados[:, i - 1] * (1 + retorno_medio + desvio_padrao * numeros_aleatorios[:, i])

        # Criar um DataFrame com os preços simulados do estoque
        precos_simulados_df = pd.DataFrame(precos_simulados, columns=[f"Dia {i+1}" for i in range(num_dias)])
        return precos_simulados_df

    except Exception as e:
        print(f"Ocorreu um erro: {e}")
        return None

def calcular_probabilidade_acima_preco_atual(precos_simulados_df, preco_atual):
    """ Calcula a probabilidade do preço futuro estar acima do preço atual. """
    num_simulacoes = len(precos_simulados_df.columns)
    num_acima_preco_atual = sum(precos_simulados_df.iloc[-1, :] > preco_atual)
    probabilidade = num_acima_preco_atual / num_simulacoes
    return probabilidade

def main(ticker, num_simulacoes, num_dias, inicio_dados, final_dados):
   
    # Executar a simulação de Monte Carlo
    precos_simulados_df = simulacao_monte_carlo(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)

    if precos_simulados_df is not None:

        # Calcular o preço atual
        preco_atual = precos_simulados_df.iloc[-1, 0] * 1.00

        # Calcular e imprimir a probabilidade do preço futuro estar acima do preço atual
        probabilidade_acima_preco_atual = calcular_probabilidade_acima_preco_atual(precos_simulados_df, preco_atual)
        #print()
        print(f"Probabilidade do preço futuro estar acima do preço atual: {probabilidade_acima_preco_atual:.2%}")
        precos_simulados = precos_simulados_df.max().mean()
    return probabilidade_acima_preco_atual, precos_simulados

if __name__ == "__main__":
    ticker = ['SBSP3.SA', 'EMBR3.SA', 'PETZ3.SA', 'STBP3.SA']
    inicio_dados = '2024-01-01'
    final_dados = '2024-05-01'
    num_simulacoes = 1000
    num_dias = 60
    probabilidade_acima_preco_atual, precos_simulados_max = main(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)
    print()
    print(f"Probabilidade do preço futuro estar acima do preço atual: {probabilidade_acima_preco_atual:.2%}")
    
    
    
