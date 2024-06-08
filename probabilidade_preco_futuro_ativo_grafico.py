import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

def simulacao_monte_carlo(ticker, num_simulacoes, num_dias, inicio_dados, final_dados):
    try:
        # Obter dados históricos do estoque
        dados_estoque = yf.download(ticker, start=inicio_dados, end=final_dados)['Close']

        # Calcular retornos diários
        retornos_diarios = dados_estoque.pct_change().dropna()

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
    """
    Calcula a probabilidade do preço futuro estar acima do preço atual.

    Args:
        precos_simulados_df (pandas.DataFrame): DataFrame contendo os preços simulados do estoque.
        preco_atual (float): O preço atual do estoque.

    Returns:
        float: A probabilidade do preço futuro estar acima do preço atual.
    """
    num_simulacoes = len(precos_simulados_df)
    num_acima_preco_atual = sum(precos_simulados_df.iloc[:, -1] > preco_atual)
    probabilidade = num_acima_preco_atual / num_simulacoes
    return probabilidade

def main():
    Ativos = ['AZUL4.SA', 'MGLU3.SA', 'GRND3.SA', 'VIVT3.SA']
    ticker = Ativos[0]
    inicio_dados = '2023-01-01'
    final_dados = '2023-05-01'
    num_simulacoes = 1000
    num_dias = 60

    # Executar a simulação de Monte Carlo
    precos_simulados_df = simulacao_monte_carlo(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)

    if precos_simulados_df is not None:
        # Calcular o preço atual
        preco_atual = precos_simulados_df.iloc[0, 0] * 1.00

        # Calcular e imprimir a probabilidade do preço futuro estar acima do preço atual após 90 dias
        probabilidade_acima_preco_atual = calcular_probabilidade_acima_preco_atual(precos_simulados_df, preco_atual)
        print(f"Probabilidade do preço futuro estar acima do preço atual após 60 dias: {probabilidade_acima_preco_atual:.2%}")
    
        # Plotar os preços simulados
        plt.figure(figsize=(12, 6))
        plt.plot(precos_simulados_df.T, color='gray', alpha=0.1)
        plt.title(f'Simulações de Monte Carlo para {ticker}')
        plt.xlabel('Dias')
        plt.ylabel('Preço')
        plt.show()

        # Plotar a distribuição dos preços futuros
        plt.figure(figsize=(10, 6))
        plt.hist(precos_simulados_df.iloc[:, -1], bins=50, color='blue', edgecolor='black')
        plt.axvline(preco_atual * 1.0, color='red', linestyle='dashed', linewidth=2, label='Preço Atual + x%')
        plt.title('Distribuição dos Preços Futuros Simulados')
        plt.xlabel('Preço Futuro')
        plt.ylabel('Frequência')
        plt.legend()
        plt.show()
    
if __name__ == "__main__":
    main()
   
   
   