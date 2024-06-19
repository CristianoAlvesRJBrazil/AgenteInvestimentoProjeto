import numpy as np
import pandas as pd
import yfinance as yf

lista_indicador = ['PETR4.SA']
inicio_dados = '2023-01-02'
final_dados = '2023-05-02'
ibov = yf.download(lista_indicador, start=inicio_dados, end=final_dados)['Close']
ibov




