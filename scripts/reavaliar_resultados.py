"""Reavalia os 30 experimentos históricos (Resultados/resultadosNN.csv) com a lógica corrigida.

O experimento original: carteiras selecionadas com dados de 2023-01-01 a
2023-05-01, probabilidade de alta estimada por Monte Carlo, e o "acerto"
verificado contra o retorno real de 2023-05-01 a 2023-08-01.

O veredito antigo tinha um bug de precedência (`if simulado and real > 1.00`)
que declarava acerto sempre que o retorno real era positivo. Este script
recalcula tudo:

- probabilidade mediana registrada em cada CSV (a previsão do agente);
- retorno real da carteira na janela futura, em duas versões:
  * cesta original (1 ação de cada papel = média de preços, fiel ao que o
    agente simulava) — no fim da janela e no pico da janela;
  * buy-and-hold com pesos monetários iguais (metodologia corrigida);
- Ibovespa no mesmo período;
- veredito honesto: direção prevista vs. direção realizada;
- teste binomial da taxa de acerto agregada contra o acaso.

Uso:
    .venv/bin/python scripts/reavaliar_resultados.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from agente_investimento import dados  # noqa: E402
from agente_investimento.backtest import (  # noqa: E402
    avaliar_previsao,
    ler_resultado_legado,
    teste_binomial_acertos,
)
from agente_investimento.carteira import Carteira, retorno_periodo_buy_and_hold  # noqa: E402

SELECAO_INICIO = "2023-01-01"
SELECAO_FIM = "2023-05-01"
AVALIACAO_FIM = "2023-08-01"


def reavaliar_arquivo(caminho: Path) -> dict | None:
    tickers, probabilidades = ler_resultado_legado(caminho)
    prob_mediana = float(probabilidades.median())

    # Preço "atual" da cesta: fim da janela de seleção (como no original).
    precos_selecao = dados.baixar_precos(tickers, SELECAO_INICIO, SELECAO_FIM)
    precos_futuro = dados.baixar_precos(tickers, SELECAO_FIM, AVALIACAO_FIM)

    completos = sorted(precos_selecao.columns) == sorted(tickers) and sorted(
        precos_futuro.columns
    ) == sorted(tickers)
    if not completos:
        com_dados = set(precos_selecao.columns) & set(precos_futuro.columns)
        faltam = sorted(set(tickers) - com_dados)
        return {
            "arquivo": caminho.name,
            "carteira": ", ".join(tickers),
            "status": f"dados incompletos ({', '.join(faltam)})",
        }

    # Cesta original: 1 ação de cada papel == média dos preços.
    cesta_atual = float(precos_selecao[tickers].mean(axis=1).iloc[-1])
    cesta_futura = precos_futuro[tickers].mean(axis=1)
    retorno_final_cesta = float(cesta_futura.iloc[-1] / cesta_atual)
    retorno_pico_cesta = float(cesta_futura.max() / cesta_atual)

    # Metodologia corrigida: buy-and-hold com pesos monetários iguais.
    retorno_equal_weight = retorno_periodo_buy_and_hold(precos_futuro, Carteira(tickers))

    return {
        "arquivo": caminho.name,
        "carteira": ", ".join(t.replace(".SA", "") for t in tickers),
        "status": "ok",
        "prob_mediana": prob_mediana,
        "retorno_final_cesta": retorno_final_cesta,
        "retorno_pico_cesta": retorno_pico_cesta,
        "retorno_equal_weight": retorno_equal_weight,
        "acertou_direcao": avaliar_previsao(prob_mediana, retorno_final_cesta),
        "acertou_direcao_pico": avaliar_previsao(prob_mediana, retorno_pico_cesta),
    }


def main() -> None:
    arquivos = sorted((RAIZ / "Resultados").glob("resultados*.csv"))
    if not arquivos:
        print("Nenhum resultado histórico encontrado em Resultados/.")
        return

    print(f"Reavaliando {len(arquivos)} experimentos históricos "
          f"(seleção {SELECAO_INICIO} a {SELECAO_FIM}, avaliação até {AVALIACAO_FIM})...\n")

    linhas = [linha for arq in arquivos if (linha := reavaliar_arquivo(arq)) is not None]
    tabela = pd.DataFrame(linhas)

    ibov = dados.baixar_benchmark(SELECAO_FIM, AVALIACAO_FIM)
    retorno_ibov = float(ibov.iloc[-1] / ibov.iloc[0])

    avaliadas = tabela[tabela["status"] == "ok"].copy()
    if not avaliadas.empty:
        avaliadas["bateu_ibov"] = avaliadas["retorno_final_cesta"] > retorno_ibov

    saida = RAIZ / "Resultados" / "reavaliacao_backtesting.csv"
    tabela_completa = tabela.merge(
        avaliadas[["arquivo", "bateu_ibov"]], on="arquivo", how="left"
    )
    tabela_completa.to_csv(saida, index=False)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        colunas_exibir = [c for c in tabela_completa.columns if c != "arquivo"]
        print(tabela_completa[colunas_exibir].round(4).to_string(index=False))

    print(f"\nRetorno do Ibovespa no período de avaliação: {retorno_ibov:.4f}")

    if not avaliadas.empty:
        total = len(avaliadas)
        for coluna, rotulo in [
            ("acertou_direcao", "retorno no FIM da janela (comparável à previsão)"),
            ("acertou_direcao_pico", "retorno no PICO da janela (métrica original, otimista)"),
        ]:
            acertos = int(avaliadas[coluna].sum())
            p_valor = teste_binomial_acertos(acertos, total)
            print(f"\nTaxa de acerto — {rotulo}: {acertos}/{total} = {acertos/total:.1%} "
                  f"(p-valor binomial vs. acaso: {p_valor:.4f})")
        print(f"Carteiras que bateram o Ibovespa: {int(avaliadas['bateu_ibov'].sum())}/{total}")
        print(
            "\nATENÇÃO: todos os experimentos compartilham a MESMA janela de avaliação "
            f"({SELECAO_FIM} a {AVALIACAO_FIM}); as observações não são independentes e o "
            "p-valor acima superestima a significância. Num trimestre em que o Ibovespa "
            "subiu, qualquer carteira comprada tende a subir junto. A validação correta "
            "usa janelas distintas: scripts/rodar_backtest.py (walk-forward)."
        )

    ignoradas = tabela[tabela["status"] != "ok"]
    if not ignoradas.empty:
        print(f"\nExperimentos sem dados completos (excluídos da taxa): {len(ignoradas)}")

    print(f"\nTabela salva em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
