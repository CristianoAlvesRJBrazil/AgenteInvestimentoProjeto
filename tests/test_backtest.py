import pandas as pd
import pytest

# Alias no import: sem ele o pytest coletaria "teste_binomial_acertos" como teste.
from agente_investimento.backtest import teste_binomial_acertos as binomial_acertos
from agente_investimento.backtest import (
    avaliar_previsao,
    gerar_janelas,
    ler_resultado_legado,
    resumir_walk_forward,
)


class TestAvaliarPrevisao:
    """Tabela-verdade do veredito — a correção do bug central do projeto antigo.

    O código original (`if retorno_simulado and retorno_real > 1.00`) declarava
    acerto sempre que o retorno real era positivo, ignorando a previsão.
    """

    def test_previu_alta_e_subiu(self):
        assert avaliar_previsao(probabilidade_alta=0.8, retorno_acumulado_real=1.10) is True

    def test_previu_alta_e_caiu_agora_conta_como_erro(self):
        # No código antigo este caso contava como acerto se real > 1; e o caso
        # de queda prevista + queda real contava como erro. Ambos errados.
        assert avaliar_previsao(probabilidade_alta=0.8, retorno_acumulado_real=0.90) is False

    def test_previu_queda_e_caiu(self):
        assert avaliar_previsao(probabilidade_alta=0.3, retorno_acumulado_real=0.95) is True

    def test_previu_queda_e_subiu(self):
        assert avaliar_previsao(probabilidade_alta=0.3, retorno_acumulado_real=1.05) is False

    def test_limiar_customizado(self):
        assert avaliar_previsao(0.55, 1.10, limiar=0.6) is False  # 0.55 < 0.6: previu queda
        assert avaliar_previsao(0.55, 0.90, limiar=0.6) is True


class TestTesteBinomial:
    def test_acerto_perfeito_tem_p_valor_baixo(self):
        assert binomial_acertos(30, 30) < 1e-6

    def test_acerto_ao_acaso_tem_p_valor_alto(self):
        assert binomial_acertos(15, 30) > 0.4

    def test_sem_amostras_retorna_nan(self):
        import math
        assert math.isnan(binomial_acertos(0, 0))


class TestLerResultadoLegado:
    def test_parseia_cabecalho_e_probabilidades(self, tmp_path):
        arquivo = tmp_path / "resultados.csv"
        arquivo.write_text(
            '"[\'UGPA3.SA\', \'TEND3.SA\', \'SMTO3.SA\', \'CPLE3.SA\']"\n0.888\n0.895\n'
        )
        tickers, probs = ler_resultado_legado(arquivo)
        assert tickers == ["UGPA3.SA", "TEND3.SA", "SMTO3.SA", "CPLE3.SA"]
        assert list(probs) == pytest.approx([0.888, 0.895])

    def test_tickers_de_seis_caracteres_nao_quebram(self, tmp_path):
        # O fatiamento fixo antigo (title_column[2:10]) quebraria com BPAC11.SA.
        arquivo = tmp_path / "resultados.csv"
        arquivo.write_text(
            '"[\'BPAC11.SA\', \'IGTI11.SA\', \'VALE3.SA\', \'ITUB4.SA\']"\n0.7\n'
        )
        tickers, _ = ler_resultado_legado(arquivo)
        assert tickers == ["BPAC11.SA", "IGTI11.SA", "VALE3.SA", "ITUB4.SA"]


class TestGerarJanelas:
    def test_janelas_do_experimento_original(self):
        # Replica o desenho original: seleção jan-mai/2023, avaliação mai-ago/2023.
        janelas = gerar_janelas("2023-01-01", "2023-08-01", 4, 3, 3)
        assert len(janelas) == 1
        assert janelas[0].selecao_fim == pd.Timestamp("2023-05-01")
        assert janelas[0].avaliacao_fim == pd.Timestamp("2023-08-01")

    def test_walk_forward_avanca_pelo_passo(self):
        janelas = gerar_janelas("2022-01-01", "2024-01-01", 4, 3, 3)
        assert len(janelas) > 1
        assert janelas[1].selecao_inicio == pd.Timestamp("2022-04-01")
        assert all(j.avaliacao_fim <= pd.Timestamp("2024-01-01") for j in janelas)

    def test_periodo_curto_gera_lista_vazia(self):
        assert gerar_janelas("2023-01-01", "2023-03-01", 4, 3, 3) == []


def test_resumir_walk_forward():
    resultado = pd.DataFrame({
        "acertou_direcao": [True, True, False, True],
        "bateu_ibov": [True, False, False, True],
        "retorno_real": [1.10, 1.05, 0.95, 1.20],
        "retorno_ibov": [1.02, 1.08, 1.01, 1.03],
    })
    resumo = resumir_walk_forward(resultado)
    assert resumo["janelas"] == 4
    assert resumo["taxa_acerto_direcao"] == pytest.approx(0.75)
    assert resumo["taxa_bateu_ibov"] == pytest.approx(0.5)
    assert 0.0 <= resumo["p_valor_binomial"] <= 1.0


def test_resumir_walk_forward_vazio():
    assert resumir_walk_forward(pd.DataFrame()) == {}
