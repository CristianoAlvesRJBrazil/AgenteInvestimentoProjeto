import numpy as np
import pandas as pd
import pytest

from agente_investimento.carteira import (
    Carteira,
    retorno_periodo_buy_and_hold,
    retornos_carteira,
    valor_buy_and_hold,
)


def test_pesos_iguais_por_padrao():
    cart = Carteira(["A", "B", "C", "D"])
    assert np.allclose(cart.pesos, [0.25, 0.25, 0.25, 0.25])


def test_pesos_sao_normalizados():
    cart = Carteira(["A", "B"], pesos=[2.0, 2.0])
    assert np.allclose(cart.pesos, [0.5, 0.5])


def test_pesos_negativos_rejeitados():
    with pytest.raises(ValueError):
        Carteira(["A", "B"], pesos=[1.5, -0.5])


def test_numero_de_pesos_diferente_rejeitado():
    with pytest.raises(ValueError):
        Carteira(["A", "B"], pesos=[1.0])


def test_retornos_carteira_media_ponderada():
    retornos = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.20]})
    cart = Carteira(["A", "B"], pesos=[0.75, 0.25])
    serie = retornos_carteira(retornos, cart)
    assert serie.iloc[0] == pytest.approx(0.075)
    assert serie.iloc[1] == pytest.approx(0.05)


def test_buy_and_hold_nao_rebalanceia():
    # A dobra, B fica parado: com 50/50 inicial o valor final é 1.5x.
    precos = pd.DataFrame({"A": [10.0, 20.0], "B": [100.0, 100.0]})
    cart = Carteira(["A", "B"])
    valor = valor_buy_and_hold(precos, cart, valor_inicial=1.0)
    assert valor.iloc[0] == pytest.approx(1.0)
    assert valor.iloc[-1] == pytest.approx(1.5)
    assert retorno_periodo_buy_and_hold(precos, cart) == pytest.approx(1.5)


def test_buy_and_hold_papel_caro_nao_domina():
    # No método antigo (média de preços = 1 ação de cada), o papel de R$100
    # dominava. Com pesos monetários iguais, queda de 50% no caro e alta de
    # 100% no barato se compensam além do esperado: 0.5*0.5 + 0.5*2 = 1.25.
    precos = pd.DataFrame({"CARO": [100.0, 50.0], "BARATO": [1.0, 2.0]})
    cart = Carteira(["CARO", "BARATO"])
    assert retorno_periodo_buy_and_hold(precos, cart) == pytest.approx(1.25)


def test_ticker_ausente_gera_erro():
    precos = pd.DataFrame({"A": [1.0, 2.0]})
    with pytest.raises(KeyError):
        valor_buy_and_hold(precos, Carteira(["A", "B"]))
