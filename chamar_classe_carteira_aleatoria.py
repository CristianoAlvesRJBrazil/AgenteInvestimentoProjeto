import carteira_aleatoria_otima_orientado_objeto
import lista_ativos_setores


def main(lista_ativos, inicio_dados, final_dados, valor_desejado):
    
    # Criação do objeto
    portfolio = carteira_aleatoria_otima_orientado_objeto.PortfolioOptimization(lista_ativos, inicio_dados, final_dados)

    # Otimização da carteira
    carteira_vencedora = portfolio.otimizar_carteira(valor_desejado)
    
    return carteira_vencedora

if __name__ == "__main__":
    
    # Lista de ativos da B3
    #lista_ativos = lista_ativos_setores.lista
    lista_ativos = lista_ativos_setores.lista_atual_ibxx_02

    # Intervalo de dados
    inicio_dados = '2024-01-01'
    final_dados = '2024-05-01'
    valor_desejado = 1.24
    
    main(lista_ativos, inicio_dados, final_dados, valor_desejado)

