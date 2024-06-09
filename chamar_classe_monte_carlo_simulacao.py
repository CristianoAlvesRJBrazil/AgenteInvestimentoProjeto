# Importação das bibliotecas necessárias
import lista_ativos_setores
import carteira_aleatoria_otima_orientado_objeto
import MonteCarloSimulationCarteiraOB

# Intervalo de dados e retorno acumulado desejando
inicio_dados = '2024-01-01'
final_dados = '2024-05-01'
valor_desejado = 1.24

# Acessar a lista de ativos e escolher a carteira otimizada    
lista_ativos = lista_ativos_setores.lista_atual_ibxx_02
portfolio = carteira_aleatoria_otima_orientado_objeto.PortfolioOptimization(lista_ativos, inicio_dados, final_dados)
carteira_vencedora = portfolio.otimizar_carteira(valor_desejado)

# Chamar a classe MonteCarloSimulationCarteiraOB
ticker =  carteira_vencedora
num_simulacoes = 1000
num_dias = 60
MonteCarloSimulationCarteiraOB.main(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)

# Carteira Vencedora = ['RRRP3.SA' 'EMBR3.SA' 'PSSA3.SA' 'CIEL3.SA']


