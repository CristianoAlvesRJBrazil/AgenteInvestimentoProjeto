import numpy as np
import pandas as pd
import pytest

from agente_investimento import dados


def _frame_multiindex(tickers, num_dias=5):
    """Imita o formato novo do yfinance: colunas MultiIndex (Price, Ticker)."""
    datas = pd.bdate_range("2023-01-02", periods=num_dias)
    colunas = pd.MultiIndex.from_product([["Close", "Open"], tickers])
    valores = np.arange(1.0, num_dias * len(colunas) + 1).reshape(num_dias, len(colunas))
    return pd.DataFrame(valores, index=datas, columns=colunas)


def test_extrair_fechamento_multiindex():
    bruto = _frame_multiindex(["AAAA3.SA", "BBBB3.SA"])
    fechamento = dados._extrair_fechamento(bruto, ["AAAA3.SA", "BBBB3.SA"])
    assert list(fechamento.columns) == ["AAAA3.SA", "BBBB3.SA"]
    assert len(fechamento) == 5


def test_extrair_fechamento_colunas_simples():
    datas = pd.bdate_range("2023-01-02", periods=3)
    bruto = pd.DataFrame({"Close": [1.0, 2.0, 3.0], "Open": [1.0, 2.0, 3.0]}, index=datas)
    fechamento = dados._extrair_fechamento(bruto, ["AAAA3.SA"])
    assert list(fechamento.columns) == ["AAAA3.SA"]


def test_extrair_fechamento_remove_timezone():
    datas = pd.bdate_range("2023-01-02", periods=3, tz="America/Sao_Paulo")
    bruto = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=datas)
    fechamento = dados._extrair_fechamento(bruto, ["AAAA3.SA"])
    assert fechamento.index.tz is None


def test_extrair_fechamento_vazio():
    assert dados._extrair_fechamento(pd.DataFrame(), ["X"]).empty


def test_calcular_retornos_simples_e_log():
    precos = pd.Series([100.0, 110.0, 99.0])
    simples = dados.calcular_retornos(precos)
    assert simples.iloc[0] == pytest.approx(0.10)
    assert simples.iloc[1] == pytest.approx(-0.10)

    log = dados.calcular_retornos(precos, log_retornos=True)
    assert log.iloc[0] == pytest.approx(np.log(1.10))


def test_baixar_precos_usa_cache_no_segundo_acesso(monkeypatch, tmp_path):
    monkeypatch.setattr(dados, "DIRETORIO_CACHE", tmp_path)
    chamadas = {"n": 0}

    def download_falso(tickers, **kwargs):
        chamadas["n"] += 1
        return _frame_multiindex(list(tickers))

    monkeypatch.setattr(dados.yf, "download", download_falso)

    precos_1 = dados.baixar_precos(["AAAA3.SA", "BBBB3.SA"], "2023-01-01", "2023-02-01")
    precos_2 = dados.baixar_precos(["AAAA3.SA", "BBBB3.SA"], "2023-01-01", "2023-02-01")

    assert chamadas["n"] == 1  # segundo acesso veio 100% do cache
    pd.testing.assert_frame_equal(precos_1, precos_2, check_freq=False)


def test_baixar_precos_descarta_ticker_sem_dados(monkeypatch, tmp_path):
    monkeypatch.setattr(dados, "DIRETORIO_CACHE", tmp_path)

    def download_falso(tickers, **kwargs):
        bruto = _frame_multiindex(list(tickers))
        bruto[("Close", "MORTO3.SA")] = np.nan  # ticker delistado: só NaN
        return bruto

    monkeypatch.setattr(dados.yf, "download", download_falso)

    precos = dados.baixar_precos(["AAAA3.SA", "MORTO3.SA"], "2023-01-01", "2023-02-01")
    assert list(precos.columns) == ["AAAA3.SA"]


def test_arquivo_cache_sanitiza_nome():
    caminho = dados._arquivo_cache("^BVSP", "2023-01-01", "2023-02-01")
    assert "^" not in caminho.name
    assert caminho.suffix == ".parquet"
