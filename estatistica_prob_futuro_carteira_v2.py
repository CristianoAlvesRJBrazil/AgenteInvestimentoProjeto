import pandas as pd
import numpy as np
import csv
import matplotlib.pyplot as plt
import probabilidade_futuro_sem_grafico as pfsg

def executar_simulacoes(ticker, num_simulacoes, num_dias, inicio_dados, final_dados):
    """
    Executa a função de probabilidade 100 vezes e armazena os resultados.

    Args:
        ticker (list): Lista de tickers para simulação.
        num_simulacoes (int): Número de simulações para executar.
        num_dias (int): Número de dias para simulação.
        inicio_dados (str): Data de início dos dados.
        final_dados (str): Data de término dos dados.

    Returns:
        list: Lista de probabilidades coletadas das simulações.
    """
    resultados = []
    for _ in range(num_simulacoes):
        probabilidade_acima_preco_atual, _ = pfsg.main(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)
        resultados.append(probabilidade_acima_preco_atual)
    return resultados

def salvar_resultados_csv(resultados, arquivo_csv='resultados.csv'):
    """
    Salva os resultados em um arquivo CSV.

    Args:
        resultados (list): Lista de resultados a serem salvos.
        arquivo_csv (str): Nome do arquivo CSV para salvar os resultados.
    """
    with open(arquivo_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['Probabilidade'])  # Escreve o cabeçalho
        for resultado in resultados:
            writer.writerow([resultado])

def ler_dados_csv(arquivo_csv='resultados.csv'):
    """
    Lê os dados do arquivo CSV e os armazena em um DataFrame.

    Args:
        arquivo_csv (str): Nome do arquivo CSV para leitura dos dados.

    Returns:
        pandas.DataFrame: DataFrame contendo os dados do CSV.
    """
    return pd.read_csv(arquivo_csv)

def gerar_resumo_estatistico(df):
    """
    Gera e imprime o resumo estatístico dos dados.

    Args:
        df (pandas.DataFrame): DataFrame contendo os dados.
    """
    print(f"Resumo estatístico:\n{df.describe()}")
    print(f"Média: {round(df['Probabilidade'].mean(), 4)}")
    print(f"Mediana: {round(df['Probabilidade'].median(), 4)}")
    print(f"Moda:\n{df['Probabilidade'].mode()}")
    print(f"Probabilidade do preço futuro estar acima do esperado: {round(df['Probabilidade'].median(), 4) * 100}%")
   
def gerar_histograma(df, bins=20):
    """
    Gera e exibe um histograma dos resultados.

    Args:
        df (pandas.DataFrame): DataFrame contendo os dados.
        bins (int): Número de bins para o histograma.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(df['Probabilidade'], bins=bins, color='skyblue', edgecolor='black')
    plt.title('Histograma das Probabilidades de Preços Futuros')
    plt.xlabel('Probabilidade')
    plt.ylabel('Frequência')
    plt.grid(True)
    plt.show()

def main():
    resultados = executar_simulacoes(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)
    salvar_resultados_csv(resultados)
    df = ler_dados_csv()
    gerar_resumo_estatistico(df)
    gerar_histograma(df)

if __name__ == "__main__":
    
    # Configurações para a simulação
    ticker = ['UGPA3.SA', 'TEND3.SA', 'CMIG4.SA', 'VIVT3.SA']
    inicio_dados = '2023-01-01'
    final_dados = '2023-05-01'
    num_simulacoes = 1000
    num_dias = 60    
    main()    
