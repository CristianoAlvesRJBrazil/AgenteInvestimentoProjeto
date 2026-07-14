"""Universo de ativos da B3: scraping dos índices com fallback para listas estáticas.

As listas estáticas abaixo vieram do projeto original (composição de ~2023/2024)
e podem conter tickers que deixaram de existir (fusões, fechamentos de capital).
Por isso todo consumo de universo deve passar por ``validar_tickers``, que
descarta o que não tem dados no período analisado.
"""

from __future__ import annotations

import logging
from io import StringIO

import pandas as pd
import requests

from . import dados

log = logging.getLogger(__name__)

URLS_INDICES = {
    "ibxx": "https://www.dadosdemercado.com.br/b3/ibxx",
    "ibov": "https://www.dadosdemercado.com.br/b3/ibov",
}

# Composição registrada no projeto original (referência histórica, pode estar defasada).
LISTA_IBXX_ESTATICA = [
    'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'PETR3.SA', 'BBAS3.SA', 'ELET3.SA', 'BBDC4.SA', 'B3SA3.SA', 'WEGE3.SA', 'ITSA4.SA',
    'ABEV3.SA', 'BPAC11.SA', 'RENT3.SA', 'EQTL3.SA', 'JBSS3.SA', 'RADL3.SA', 'SUZB3.SA', 'PRIO3.SA', 'RDOR3.SA', 'EMBR3.SA',
    'SBSP3.SA', 'RAIL3.SA', 'UGPA3.SA', 'VBBR3.SA', 'BBSE3.SA', 'GGBR4.SA', 'ENEV3.SA', 'CMIG4.SA', 'VIVT3.SA', 'HAPV3.SA',
    'BBDC3.SA', 'ASAI3.SA', 'KLBN11.SA', 'CPLE6.SA', 'BRFS3.SA', 'ENGI11.SA', 'CSAN3.SA', 'TOTS3.SA', 'TIMS3.SA', 'NTCO3.SA',
    'LREN3.SA', 'CCRO3.SA', 'HYPE3.SA', 'EGIE3.SA', 'CPLE3.SA', 'ELET6.SA', 'STBP3.SA', 'TRPL4.SA', 'SANB11.SA', 'CXSE3.SA',
    'CSNA3.SA', 'TAEE11.SA', 'SMFT3.SA', 'GOAU4.SA', 'CPFE3.SA', 'RRRP3.SA', 'CIEL3.SA', 'MULT3.SA', 'ENAT3.SA',
    'PSSA3.SA', 'CMIN3.SA', 'RECV3.SA', 'CYRE3.SA', 'CRFB3.SA', 'BRKM5.SA', 'BRAP4.SA', 'IGTI11.SA', 'POMO4.SA', 'MGLU3.SA',
    'SMTO3.SA', 'VAMO3.SA', 'CSMG3.SA', 'USIM5.SA', 'MRFG3.SA', 'AURE3.SA', 'GMAT3.SA', 'FLRY3.SA', 'SLCE3.SA', 'YDUQ3.SA',
    'COGN3.SA', 'RAIZ4.SA', 'ARZZ3.SA', 'AZUL4.SA', 'SOMA3.SA', 'DIRR3.SA', 'VIVA3.SA', 'IRBR3.SA', 'MRVE3.SA', 'ECOR3.SA',
    'DXCO3.SA', 'LWSA3.SA', 'BEEF3.SA', 'ALPA4.SA', 'TEND3.SA', 'EZTC3.SA', 'PETZ3.SA', 'PCAR3.SA', 'CVCB3.SA', 'MOVI3.SA',
]

LISTA_IBOV_ESTATICA = [
    'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'PETR3.SA', 'BBAS3.SA', 'ELET3.SA', 'BBDC4.SA', 'B3SA3.SA', 'WEGE3.SA',
    'ITSA4.SA', 'ABEV3.SA', 'BPAC11.SA', 'RENT3.SA', 'EQTL3.SA', 'JBSS3.SA', 'RADL3.SA', 'SUZB3.SA', 'PRIO3.SA',
    'RDOR3.SA', 'EMBR3.SA', 'SBSP3.SA', 'RAIL3.SA', 'UGPA3.SA', 'VBBR3.SA', 'BBSE3.SA', 'GGBR4.SA', 'ENEV3.SA',
    'CMIG4.SA', 'VIVT3.SA', 'HAPV3.SA', 'BBDC3.SA', 'ASAI3.SA', 'KLBN11.SA', 'CPLE6.SA', 'BRFS3.SA', 'ENGI11.SA',
    'CSAN3.SA', 'TOTS3.SA', 'TIMS3.SA', 'NTCO3.SA', 'LREN3.SA', 'CCRO3.SA', 'HYPE3.SA', 'EGIE3.SA', 'ELET6.SA',
    'TRPL4.SA', 'SANB11.SA', 'CSNA3.SA', 'TAEE11.SA', 'GOAU4.SA', 'CPFE3.SA', 'RRRP3.SA', 'CIEL3.SA', 'MULT3.SA',
    'CMIN3.SA', 'RECV3.SA', 'CRFB3.SA', 'CYRE3.SA', 'BRKM5.SA', 'BRAP4.SA', 'IGTI11.SA', 'MGLU3.SA', 'SMTO3.SA',
    'VAMO3.SA', 'MRFG3.SA', 'USIM5.SA', 'FLRY3.SA', 'SLCE3.SA', 'YDUQ3.SA', 'RAIZ4.SA', 'COGN3.SA', 'ARZZ3.SA',
    'AZUL4.SA', 'SOMA3.SA', 'VIVA3.SA', 'IRBR3.SA', 'MRVE3.SA', 'DXCO3.SA', 'LWSA3.SA', 'BEEF3.SA', 'ALPA4.SA',
    'EZTC3.SA', 'PETZ3.SA', 'PCAR3.SA', 'CVCB3.SA',
]

LISTAS_ESTATICAS = {"ibxx": LISTA_IBXX_ESTATICA, "ibov": LISTA_IBOV_ESTATICA}


def obter_indice_online(indice: str = "ibxx", timeout: int = 20) -> list[str]:
    """Raspa a composição atual do índice em dadosdemercado.com.br."""
    url = URLS_INDICES[indice]
    resposta = requests.get(
        url, timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
    )
    resposta.raise_for_status()

    tabelas = pd.read_html(StringIO(resposta.text))
    if not tabelas:
        raise RuntimeError(f"Nenhuma tabela encontrada em {url}.")

    tabela = tabelas[0]
    coluna_codigo = next((c for c in ("Código", "Ticker") if c in tabela.columns), None)
    if coluna_codigo is None:
        raise RuntimeError(f"Tabela de {url} sem coluna de código; layout mudou.")

    return [f"{codigo}.SA" for codigo in tabela[coluna_codigo].astype(str)]


def obter_universo(indice: str = "ibxx", online: bool = True) -> list[str]:
    """Composição do índice, tentando a fonte online e caindo para a lista estática."""
    if indice not in LISTAS_ESTATICAS:
        raise ValueError(f"Índice desconhecido: {indice!r}. Opções: {sorted(LISTAS_ESTATICAS)}")
    if online:
        try:
            return obter_indice_online(indice)
        except Exception as erro:  # rede fora, layout mudou etc.
            log.warning("Falha ao obter %s online (%s); usando lista estática.", indice, erro)
    return list(LISTAS_ESTATICAS[indice])


def validar_tickers(tickers: list[str], inicio: str, fim: str) -> list[str]:
    """Filtra tickers que têm dados de preço no período (remove delistados/renomeados)."""
    precos = dados.baixar_precos(tickers, inicio, fim)
    validos = list(precos.columns)
    removidos = sorted(set(tickers) - set(validos))
    if removidos:
        log.info("Tickers sem dados no período removidos: %s", ", ".join(removidos))
    return validos
