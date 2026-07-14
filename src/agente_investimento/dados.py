"""Camada de dados: download de preços via yfinance com cache local em parquet.

O cache evita repetir downloads do Yahoo Finance (lento e sujeito a limite de
requisições). Cada ticker/período vira um arquivo parquet em ``dados_cache/``.
Os preços usados são os de fechamento ajustado (auto_adjust=True), que
incorporam splits e dividendos — mais correto para cálculo de retornos do que
o fechamento bruto usado na versão antiga do projeto.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)

DIRETORIO_CACHE = Path(__file__).resolve().parents[2] / "dados_cache"


def _arquivo_cache(ticker: str, inicio: str, fim: str) -> Path:
    nome = re.sub(r"[^A-Za-z0-9._-]", "_", f"{ticker}_{inicio}_{fim}") + ".parquet"
    return DIRETORIO_CACHE / nome


def _extrair_fechamento(dados: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    """Normaliza o retorno do yfinance para um DataFrame com uma coluna por ticker.

    O formato de colunas do yfinance varia com a versão e com o número de
    tickers (MultiIndex ``(Price, Ticker)`` ou colunas simples); esta função
    isola o resto do pacote dessa instabilidade.
    """
    if dados is None or dados.empty:
        return pd.DataFrame()

    if isinstance(dados.columns, pd.MultiIndex):
        fechamento = dados["Close"]
        if isinstance(fechamento, pd.Series):
            fechamento = fechamento.to_frame(tickers[0])
    else:
        fechamento = dados[["Close"]].rename(columns={"Close": tickers[0]})

    if isinstance(fechamento.index, pd.DatetimeIndex) and fechamento.index.tz is not None:
        fechamento.index = fechamento.index.tz_localize(None)
    return fechamento


def baixar_precos(
    tickers: list[str] | str,
    inicio: str,
    fim: str,
    usar_cache: bool = True,
) -> pd.DataFrame:
    """Baixa preços de fechamento ajustado; retorna DataFrame com uma coluna por ticker.

    Tickers sem dados no período são descartados com aviso no log — o chamador
    deve conferir ``df.columns`` para saber quais sobreviveram.
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    colunas: dict[str, pd.Series] = {}
    faltantes: list[str] = []
    for ticker in tickers:
        arquivo = _arquivo_cache(ticker, inicio, fim)
        if usar_cache and arquivo.exists():
            colunas[ticker] = pd.read_parquet(arquivo)["preco"]
        else:
            faltantes.append(ticker)

    if faltantes:
        baixados = yf.download(
            faltantes, start=inicio, end=fim,
            auto_adjust=True, progress=False, group_by="column",
        )
        fechamento = _extrair_fechamento(baixados, faltantes)
        for ticker in faltantes:
            if ticker not in fechamento.columns:
                log.warning("Sem dados para %s entre %s e %s; ignorando.", ticker, inicio, fim)
                continue
            serie = fechamento[ticker].dropna()
            if serie.empty:
                log.warning("Dados vazios para %s entre %s e %s; ignorando.", ticker, inicio, fim)
                continue
            if usar_cache:
                DIRETORIO_CACHE.mkdir(parents=True, exist_ok=True)
                serie.to_frame("preco").to_parquet(_arquivo_cache(ticker, inicio, fim))
            colunas[ticker] = serie

    validos = [t for t in tickers if t in colunas]
    if not validos:
        return pd.DataFrame()
    return pd.concat([colunas[t].rename(t) for t in validos], axis=1).sort_index()


def calcular_retornos(precos: pd.DataFrame | pd.Series, log_retornos: bool = False):
    """Retornos diários a partir dos preços; simples por padrão, logarítmicos se pedido."""
    if log_retornos:
        retornos = np.log(precos / precos.shift(1))
    else:
        retornos = precos.pct_change(fill_method=None)
    return retornos.dropna(how="all") if isinstance(retornos, pd.DataFrame) else retornos.dropna()


def baixar_benchmark(inicio: str, fim: str, ticker: str = "^BVSP", usar_cache: bool = True) -> pd.Series:
    """Preços do índice de referência (Ibovespa por padrão)."""
    precos = baixar_precos([ticker], inicio, fim, usar_cache=usar_cache)
    if precos.empty:
        raise RuntimeError(f"Não foi possível obter dados do benchmark {ticker}.")
    return precos[ticker]
