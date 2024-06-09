# Importação das bibliotecas necessárias
import lista_ativos_setores
import carteira_aleatoria_otima_orientado_objeto
import MonteCarloSimulationCarteiraOB

# Intervalo de dados e retorno acumulado desejando
inicio_dados = '2024-01-01'
final_dados = '2024-06-01'
valor_desejado = 1.10

# Acessar a lista de ativos e escolher a carteira otimizada    
lista_ativos = lista_ativos_setores.lista_atual_ibxx_02
portfolio = carteira_aleatoria_otima_orientado_objeto.PortfolioOptimization(lista_ativos, inicio_dados, final_dados)
carteira_vencedora = portfolio.otimizar_carteira(valor_desejado)

# Chamar a classe MonteCarloSimulationCarteiraOB
ticker =  carteira_vencedora
num_simulacoes = 1000
num_dias = 60
MonteCarloSimulationCarteiraOB.main(ticker, num_simulacoes, num_dias, inicio_dados, final_dados)
# retorno_acumulado = portfolio.calcular_retorno_acumulado(carteira_vencedora[1])
print(f"Carteira vencedora = {carteira_vencedora}")
# print(f"Retorno Acumulado da Carteira: {retorno_acumulado}")

# Resultados da Predição:
# Probabilidade do preço futuro estar acima do preço atual após 60 dias: 79.10%
# Carteira vencedora: ['TEND3.SA', 'VIVA3.SA', 'SMFT3.SA', 'ECOR3.SA', 'BRKM5.SA', 'CYRE3.SA', 'BPAC11.SA', 'LREN3.SA', 'CSNA3.SA', 'BBDC3.SA']


