import numpy as np
import pandas as pd
import yfinance as yf
#import matplotlib.pyplot as plt
import tratamento_saida_carteira_aleatoria
import carteira_aleatoria_otima_orientado_objeto

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

def main(ativo):
    ticker = ativo[0]
    inicio_dados = '2024-01-29'
    final_dados = '2024-05-29'
    num_simulacoes = 1000
    num_dias = 60

    # Executar a simulação de Monte Carlo
    precos_simulados_df = simulacao_monte_carlo(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)

    if precos_simulados_df is not None:
        # Calcular o preço atual
        preco_atual = precos_simulados_df.iloc[0, 0]

        # Calcular e imprimir a probabilidade do preço futuro estar acima do preço atual após 60 dias
        probabilidade_acima_preco_atual = calcular_probabilidade_acima_preco_atual(precos_simulados_df, preco_atual * 1.00)
        print(f"Probabilidade do preço futuro estar acima do preço atual após 60 dias: {probabilidade_acima_preco_atual:.2%}")
    
    return probabilidade_acima_preco_atual   

if __name__ == "__main__":
    
    # Lista de ativos da B3
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

    # Intervalo de Datas para análise 
    inicio_dados = '2023-01-01'
    final_dados = '2023-05-01'
    valor_desejado = 1.24
    otimizador = carteira_aleatoria_otima_orientado_objeto.PortfolioOptimization(lista_ativos, inicio_dados, final_dados)
    ativo = otimizador.otimizar_carteira(valor_desejado)
    print("Carteira Vencedora =", ativo)
    main(ativo)
    
   