import numpy as np
import pandas as pd
import yfinance as yf
import lista_ativos_setores as las

class PortfolioOptimization:
    def __init__(self, lista_ativos, inicio_dados, final_dados):
        self.lista_ativos = lista_ativos
        self.inicio_dados = inicio_dados
        self.final_dados = final_dados
        self.dados_ativos = self.obter_dados_ativos()
        self.retornos_ativos = self.calcular_retornos_percentuais(self.dados_ativos)

    def obter_dados_ativos(self):
        """Obtém os dados históricos de fechamento dos ativos, ignorando os ativos que não retornarem dados."""
        dados_todos_ativos = []
        ativos_validos = []

        for ativo in self.lista_ativos:
            try:
                # Baixa os dados do ativo
                dados = yf.download(ativo, start=self.inicio_dados, end=self.final_dados)['Close']
                # Verifica se os dados retornados não estão vazios
                if dados.empty:
                    print(f'Dados para {ativo} estão vazios. Ignorando.')
                    continue

                # Renomeia a coluna para o nome do ativo
                dados.rename(ativo, inplace=True)
                # Adiciona os dados à lista
                dados_todos_ativos.append(dados)
                ativos_validos.append(ativo)
                print(f'Dados para {ativo} obtidos com sucesso.')

            except Exception as e:
                print(f'Erro ao obter dados para {ativo}: {e}')
                continue

        # Concatena todos os dados em um DataFrame
        if dados_todos_ativos:
            dados_consolidados = pd.concat(dados_todos_ativos, axis=1)
        else:
            dados_consolidados = pd.DataFrame()

        # Atualiza a lista de ativos com apenas os ativos válidos
        self.lista_ativos = ativos_validos

        return dados_consolidados

    def calcular_retornos_percentuais(self, dados):
        """Calcula os retornos percentuais diários dos ativos."""
        retornos = dados.pct_change(fill_method=None).dropna()
        return retornos

    def simular_carteiras(self, num_carteiras=1000, num_ativos=4):
        """Simula carteiras aleatórias com os ativos disponíveis."""
        carteiras = []
        for _ in range(num_carteiras):
            ativos_carteira = np.random.choice(self.lista_ativos, size=num_ativos, replace=False)
            retorno_carteira = self.retornos_ativos[ativos_carteira].mean(axis=1)
            carteiras.append((ativos_carteira, retorno_carteira))
        return carteiras

    def calcular_retorno_medio(self, retorno_carteira):
        """Calcula o retorno médio do período."""
        retorno_medio = retorno_carteira.mean()
        return retorno_medio

    def calcular_desvio_padrao(self, retorno_carteira):
        """Calcula o desvio padrão do período."""
        desvio_padrao = retorno_carteira.std()
        return desvio_padrao

    def calcular_retorno_acumulado(self, retorno_carteira):
        """Calcula o retorno acumulado do período."""
        if not retorno_carteira.empty:
            retorno_acumulado = (1 + retorno_carteira).cumprod().iloc[-1]
            return retorno_acumulado
        else:
            return None

    def encontrar_carteira_vencedora(self, carteiras, retorno_ibov):
        """Encontra a carteira com o maior retorno acumulado acima do Ibovespa."""
        retorno_ibov_medio = retorno_ibov.mean()

        def calculo(carteira):
            retorno_carteira = carteira[1]
            retorno_medio_carteira = self.calcular_retorno_medio(retorno_carteira)
            retorno_acumulado_carteira = self.calcular_retorno_acumulado(retorno_carteira)
            if retorno_acumulado_carteira is not None:
                return (retorno_medio_carteira - retorno_ibov_medio) * retorno_acumulado_carteira
            else:
                return -np.inf  # Retorna valor muito baixo para ignorar carteiras inválidas

        carteira_vencedora = max(carteiras, key=calculo)
        return carteira_vencedora

    def obter_dados_ibov(self, inicio_dados, final_dados):
        """Obtém os dados históricos de fechamento do Ibovespa."""
        lista_indicador = '^BVSP'
        ibov = yf.download(lista_indicador, start=inicio_dados, end=final_dados)['Close']
        retorno_ibov = self.calcular_retornos_percentuais(ibov)
        return retorno_ibov

    def otimizar_carteira(self, valor_desejado, max_iteracoes=10000):
        """Obtém a carteira otimizada com rendimento acima do Ibovespa."""
        iteracao_atual = 0
        carteira_vencedora = None
        retorno_ibov = self.obter_dados_ibov(self.inicio_dados, self.final_dados)

        while iteracao_atual < max_iteracoes:
            carteiras_simuladas = self.simular_carteiras()
            carteira_vencedora = self.encontrar_carteira_vencedora(carteiras_simuladas, retorno_ibov)
            retorno_acumulado_carteira = self.calcular_retorno_acumulado(carteira_vencedora[1])

            if retorno_acumulado_carteira is not None and retorno_acumulado_carteira >= valor_desejado:
                break

            iteracao_atual += 1

        if iteracao_atual == max_iteracoes:
            print(f"Não foi possível encontrar uma carteira que atendesse ao valor desejado de retorno acumulado após {max_iteracoes} iterações.")
        else:
            retorno_medio_carteira = self.calcular_retorno_medio(carteira_vencedora[1])
            retorno_medio_carteira_percentual = retorno_medio_carteira * 100
            desvio_padrao = self.calcular_desvio_padrao(carteira_vencedora[1])
            desvio_padrao_percentual = desvio_padrao * 100
            dias_uteis_ano = 252
            media_anual = ((1 + retorno_medio_carteira) ** dias_uteis_ano - 1) * 100
            desvio_padrao_anual = (desvio_padrao * np.sqrt(dias_uteis_ano)) * 100

            print("Resultados da Carteira Vencedora:")
            print(f"Carteira Vencedora = {carteira_vencedora[0]}")
            print(f"Retorno Médio Diário: {retorno_medio_carteira_percentual:.2f}%")
            print(f"Desvio Padrão Diário: {desvio_padrao_percentual:.2f}%")
            print(f"Retorno Médio Anual da Carteira: {media_anual:.2f}%")
            print(f"Desvio Padrão do Retorno Médio Anual da Carteira: {desvio_padrao_anual:.2f}%")
            print(f"Retorno Acumulado: {retorno_acumulado_carteira:.4f}")

            return carteira_vencedora[0]

if __name__ == "__main__":
    # Lista de ativos da B3
    lista_ativos = las.main()
    
    # Intervalo de Datas para análise
    inicio_dados = '2023-01-01'
    final_dados = '2023-05-01'

    valor_desejado = float(input("Informe o valor desejado de retorno acumulado: "))
    
    # Chamando o otimizador de carteira
    otimizador = PortfolioOptimization(lista_ativos, inicio_dados, final_dados)
    carteira_vencedora = otimizador.otimizar_carteira(valor_desejado)
    
    print(f"Carteira Vencedora = {carteira_vencedora}")



