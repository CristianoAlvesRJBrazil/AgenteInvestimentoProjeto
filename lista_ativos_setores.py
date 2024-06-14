import requests
from bs4 import BeautifulSoup
import pandas as pd

def main():
    
    # URL da página
    # url = 'https://www.dadosdemercado.com.br/b3/ibxx'
    # url = 'https://www.dadosdemercado.com.br/acoes'
    url = 'https://www.dadosdemercado.com.br/b3/ibov'

    # Faz a solicitação HTTP para obter o conteúdo da página
    response = requests.get(url)
    response.raise_for_status()  # Verifica se houve algum erro na requisição

    # Analisa o conteúdo HTML com BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')

    # Encontra todas as tabelas na página
    tables = soup.find_all('table')

    # Verifica se encontrou pelo menos uma tabela
    if tables:
        # Seleciona a primeira tabela
        first_table_html = str(tables[0])
        
        # Converte a tabela HTML para um DataFrame do pandas
        df = pd.read_html(first_table_html)[0]
        #lista_ativos = df['Código'].copy()
        lista_ativos = df['Código'].copy()
        lista_ativos = lista_ativos.to_list()
        # Exibe o DataFrame
        #print("Primeira Tabela:")
        #print(df)
    else:
        print("Nenhuma tabela encontrada na página.")

    # Adicionando ".SA" a cada elemento da lista
    lista_ibxx = [codigo + '.SA' for codigo in lista_ativos]
    return lista_ibxx

lista_atual_ibxx_02 = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'PETR3.SA', 'BBAS3.SA', 'ELET3.SA', 'BBDC4.SA', 'B3SA3.SA', 'WEGE3.SA', 'ITSA4.SA',
                    'ABEV3.SA', 'BPAC11.SA', 'RENT3.SA', 'EQTL3.SA', 'JBSS3.SA', 'RADL3.SA', 'SUZB3.SA', 'PRIO3.SA', 'RDOR3.SA', 'EMBR3.SA',
                    'SBSP3.SA', 'RAIL3.SA', 'UGPA3.SA', 'VBBR3.SA', 'BBSE3.SA', 'GGBR4.SA', 'ENEV3.SA', 'CMIG4.SA', 'VIVT3.SA', 'HAPV3.SA',
                    'BBDC3.SA', 'ASAI3.SA', 'KLBN11.SA', 'CPLE6.SA', 'BRFS3.SA', 'ENGI11.SA', 'CSAN3.SA', 'TOTS3.SA', 'TIMS3.SA', 'NTCO3.SA',
                    'LREN3.SA', 'CCRO3.SA', 'HYPE3.SA', 'EGIE3.SA', 'CPLE3.SA', 'ELET6.SA', 'STBP3.SA', 'TRPL4.SA', 'SANB11.SA',
                    'CXSE3.SA', 'CSNA3.SA', 'TAEE11.SA', 'SMFT3.SA', 'GOAU4.SA', 'CPFE3.SA', 'RRRP3.SA', 'CIEL3.SA', 'MULT3.SA', 'ENAT3.SA',
                    'PSSA3.SA', 'CMIN3.SA', 'RECV3.SA', 'CYRE3.SA', 'CRFB3.SA', 'BRKM5.SA', 'BRAP4.SA', 'IGTI11.SA', 'POMO4.SA', 'MGLU3.SA',
                    'SMTO3.SA', 'VAMO3.SA', 'CSMG3.SA', 'USIM5.SA', 'MRFG3.SA', 'AURE3.SA', 'GMAT3.SA', 'FLRY3.SA', 'SLCE3.SA', 'YDUQ3.SA',
                    'COGN3.SA', 'RAIZ4.SA', 'ARZZ3.SA', 'AZUL4.SA', 'SOMA3.SA', 'DIRR3.SA', 'VIVA3.SA', 'IRBR3.SA', 'MRVE3.SA', 'ECOR3.SA',
                    'DXCO3.SA', 'LWSA3.SA', 'BEEF3.SA', 'ALPA4.SA', 'TEND3.SA', 'EZTC3.SA', 'PETZ3.SA', 'PCAR3.SA', 'CVCB3.SA', 'MOVI3.SA']


lista_ibxl = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'PETR3.SA', 'BBAS3.SA', 'ELET3.SA', 'BBDC4.SA', 'B3SA3.SA', 'WEGE3.SA', 'ITSA4.SA', 'ABEV3.SA', 'BPAC11.SA', 'RENT3.SA', 'EQTL3.SA', 'JBSS3.SA', 'RADL3.SA', 'SUZB3.SA', 'PRIO3.SA', 'RDOR3.SA', 'EMBR3.SA', 'SBSP3.SA', 'RAIL3.SA', 'UGPA3.SA', 'VBBR3.SA', 'BBSE3.SA', 'GGBR4.SA', 'CMIG4.SA', 'HAPV3.SA', 'ASAI3.SA', 'KLBN11.SA', 'CPLE6.SA', 'BRFS3.SA', 'CSAN3.SA', 'TOTS3.SA', 'TIMS3.SA', 'NTCO3.SA', 'LREN3.SA', 'HYPE3.SA', 'CSNA3.SA', 'RRRP3.SA', 'CIEL3.SA', 'MULT3.SA', 'CYRE3.SA', 'MGLU3.SA', 'USIM5.SA', 'ARZZ3.SA', 'AZUL4.SA', 'SOMA3.SA', 'MRVE3.SA']

lista_ibov = ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'PETR3.SA', 'BBAS3.SA', 'ELET3.SA', 'BBDC4.SA', 'B3SA3.SA', 'WEGE3.SA',
              'ITSA4.SA', 'ABEV3.SA', 'BPAC11.SA', 'RENT3.SA', 'EQTL3.SA', 'JBSS3.SA', 'RADL3.SA', 'SUZB3.SA', 'PRIO3.SA',
              'RDOR3.SA', 'EMBR3.SA', 'SBSP3.SA', 'RAIL3.SA', 'UGPA3.SA', 'VBBR3.SA', 'BBSE3.SA', 'GGBR4.SA', 'ENEV3.SA',
              'CMIG4.SA', 'VIVT3.SA', 'HAPV3.SA', 'BBDC3.SA', 'ASAI3.SA', 'KLBN11.SA', 'CPLE6.SA', 'BRFS3.SA', 'ENGI11.SA',
              'CSAN3.SA', 'TOTS3.SA', 'TIMS3.SA', 'NTCO3.SA', 'LREN3.SA', 'CCRO3.SA', 'HYPE3.SA', 'EGIE3.SA', 'ELET6.SA',
              'TRPL4.SA', 'SANB11.SA', 'CSNA3.SA', 'TAEE11.SA', 'GOAU4.SA', 'CPFE3.SA', 'RRRP3.SA', 'CIEL3.SA', 'MULT3.SA',
              'CMIN3.SA', 'RECV3.SA', 'CRFB3.SA', 'CYRE3.SA', 'BRKM5.SA', 'BRAP4.SA', 'IGTI11.SA', 'MGLU3.SA', 'SMTO3.SA',
              'VAMO3.SA', 'MRFG3.SA', 'USIM5.SA', 'FLRY3.SA', 'SLCE3.SA', 'YDUQ3.SA', 'RAIZ4.SA', 'COGN3.SA', 'ARZZ3.SA',
              'AZUL4.SA', 'SOMA3.SA', 'VIVA3.SA', 'IRBR3.SA', 'MRVE3.SA', 'DXCO3.SA', 'LWSA3.SA', 'BEEF3.SA', 'ALPA4.SA',
              'EZTC3.SA', 'PETZ3.SA', 'PCAR3.SA', 'CVCB3.SA']

if __name__ == "__main__":
    lista_atual_ibxx_01 = main()
    print(lista_atual_ibxx_01)


