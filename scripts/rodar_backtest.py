"""Backtest walk-forward: valida a estratégia fora da amostra, janela após janela.

Em cada janela o agente seleciona a carteira SÓ com dados passados, prevê a
direção via Monte Carlo e é julgado pelo que aconteceu na janela seguinte.

Exemplos:
    .venv/bin/python scripts/rodar_backtest.py
    .venv/bin/python scripts/rodar_backtest.py --inicio 2022-01-01 --fim 2024-07-01 \
        --criterio excesso_ibov --num-carteiras 2000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from agente_investimento import universo  # noqa: E402
from agente_investimento.backtest import resumir_walk_forward, rodar_walk_forward  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest walk-forward da estratégia")
    parser.add_argument("--inicio", default="2022-01-01")
    parser.add_argument("--fim", default="2024-07-01")
    parser.add_argument("--indice", default="ibov", choices=["ibxx", "ibov"])
    parser.add_argument("--offline", action="store_true", help="usar lista estática do índice")
    parser.add_argument("--meses-selecao", type=int, default=4)
    parser.add_argument("--meses-avaliacao", type=int, default=3)
    parser.add_argument("--passo-meses", type=int, default=3)
    parser.add_argument("--num-carteiras", type=int, default=1000)
    parser.add_argument("--num-ativos", type=int, default=4)
    parser.add_argument("--num-simulacoes", type=int, default=5000)
    parser.add_argument("--criterio", default="sharpe",
                        choices=["sharpe", "sortino", "retorno_acumulado", "excesso_ibov"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tickers = universo.obter_universo(args.indice, online=not args.offline)
    print(f"Universo {args.indice.upper()}: {len(tickers)} tickers. "
          f"Walk-forward de {args.inicio} a {args.fim} "
          f"(seleção {args.meses_selecao}m, avaliação {args.meses_avaliacao}m, "
          f"passo {args.passo_meses}m)...\n")

    resultado = rodar_walk_forward(
        universo=tickers,
        inicio=args.inicio,
        fim=args.fim,
        meses_selecao=args.meses_selecao,
        meses_avaliacao=args.meses_avaliacao,
        passo_meses=args.passo_meses,
        num_carteiras=args.num_carteiras,
        num_ativos=args.num_ativos,
        num_simulacoes=args.num_simulacoes,
        criterio=args.criterio,
        seed=args.seed,
    )

    if resultado.empty:
        print("Nenhuma janela avaliável no período.")
        return

    import pandas as pd
    with pd.option_context("display.width", 220, "display.max_columns", None):
        print(resultado.round(4).to_string(index=False))

    resumo = resumir_walk_forward(resultado)
    print("\n" + "=" * 70)
    print("RESUMO DO WALK-FORWARD (tudo out-of-sample)")
    print("=" * 70)
    print(f"Janelas avaliadas:               {resumo['janelas']}")
    print(f"Taxa de acerto de direção:       {resumo['taxa_acerto_direcao']:.1%} "
          f"(p-valor vs. acaso: {resumo['p_valor_binomial']:.4f})")
    print(f"Carteiras que bateram o IBOV:    {resumo['taxa_bateu_ibov']:.1%}")
    print(f"Retorno médio das carteiras:     {resumo['retorno_medio_carteiras']:.4f}")
    print(f"Retorno médio do IBOV:           {resumo['retorno_medio_ibov']:.4f}")

    saida = RAIZ / "Resultados" / "walk_forward.csv"
    resultado.to_csv(saida, index=False)
    print(f"\nDetalhe por janela salvo em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
