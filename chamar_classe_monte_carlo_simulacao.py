# Importação das bibliotecas necessárias
import lista_ativos_setores
import carteira_aleatoria_otima_orientado_objeto
import MonteCarloSimulationCarteiraOB
import MonteCarloSimulationV2

def main(inicio_dados, final_dados, valor_desejado):
    # Acessar a lista de ativos e escolher a carteira otimizada    
    lista_ativos = lista_ativos_setores.lista_ibov
    portfolio = carteira_aleatoria_otima_orientado_objeto.PortfolioOptimization(lista_ativos, inicio_dados, final_dados)
    carteira_vencedora = portfolio.otimizar_carteira(valor_desejado)

    # Chamar as classes MonteCarloSimulation
    ticker = carteira_vencedora
    num_simulacoes = 1000
    num_dias = 60
    MonteCarloSimulationCarteiraOB.main(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)
    
    # Imprimir Gráfico da simulação de Monte Carlo com histórico passado no gráfico
    MonteCarloSimulationV2.main(ticker, inicio_dados, final_dados)

    # retorno_acumulado = portfolio.calcular_retorno_acumulado(carteira_vencedora[1])
    print(f"Carteira vencedora = {carteira_vencedora}")
    # print(f"Retorno Acumulado da Carteira: {retorno_acumulado}")

    # Resultados da Predição:
    # Probabilidade do preço futuro estar acima do preço atual após 60 dias: 92.50%
    # Carteira vencedora = ['ENAT3.SA', 'STBP3.SA', 'PETR3.SA', 'RRRP3.SA']
    pass
    
if __name__ == "__main__":
    # Intervalo de dados e retorno acumulado desejando
    inicio_dados = '2023-01-02'
    final_dados = '2023-05-02'
    valor_desejado = 1.24
    main(inicio_dados, final_dados, valor_desejado)

