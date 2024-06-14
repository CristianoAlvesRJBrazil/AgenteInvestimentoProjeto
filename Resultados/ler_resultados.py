import pandas as pd
import os

# Nome do arquivo CSV a ser lido
filename = '/home/cristianoalves/Agente/MultiAgenteArtigo/Resultados/resultados15.csv'

# Verifica se o arquivo existe
if not os.path.isfile(filename):
    print(f"Erro: O arquivo '{filename}' não foi encontrado.")
else:
    try:
        # Ler o arquivo CSV para um DataFrame
        df = pd.read_csv(filename)

        # Mostrar as primeiras linhas do DataFrame
        print(df)
     
    except Exception as e:
        print(f"Ocorreu um erro ao ler o arquivo: {e}")
        
# Resumo estatístico
resumo_estatistico = df.describe()
print(f"Resumo estatístico:\n {resumo_estatistico}")
    
# Calcular a média
media = df.mean().iloc[0]
print(f"A média é: {round(media, 4)}")

# Calcular a mediana
mediana = df.median().iloc[0]
print(f"A mediana é: {round(mediana, 4)}")

# Calcular a moda
moda = df.mode().iloc[0, 0]
print(f"A moda é: {round(moda, 4)}")

print(f"Probabilidade do preço futuro estar acima do esperado: {round(mediana, 4)*100}%")
print(f"Carteira_Vencedora = {resumo_estatistico.head(1)}")
