"""Pipeline completo: universo -> seleção de carteira -> Monte Carlo -> relatório.

Exemplos:
    .venv/bin/python scripts/rodar_pipeline.py
    .venv/bin/python scripts/rodar_pipeline.py --inicio 2024-01-02 --fim 2024-05-01 \
        --num-carteiras 2000 --num-simulacoes 10000 --horizonte 60 --criterio sharpe
    .venv/bin/python scripts/rodar_pipeline.py --otimizar-pesos   # pesos via SLSQP
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

from agente_investimento import dados, otimizacao, universo  # noqa: E402
from agente_investimento.monte_carlo import SimulacaoMonteCarlo  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline de seleção e simulação de carteira")
    parser.add_argument("--inicio", default="2023-01-01", help="início da janela de análise")
    parser.add_argument("--fim", default="2023-05-01", help="fim da janela de análise")
    parser.add_argument("--indice", default="ibxx", choices=["ibxx", "ibov"])
    parser.add_argument("--offline", action="store_true", help="não raspar índice online")
    parser.add_argument("--num-carteiras", type=int, default=1000)
    parser.add_argument("--num-ativos", type=int, default=4)
    parser.add_argument("--num-simulacoes", type=int, default=10000)
    parser.add_argument("--horizonte", type=int, default=60, help="dias úteis simulados")
    parser.add_argument("--criterio", default="sharpe",
                        choices=["sharpe", "sortino", "retorno_acumulado", "excesso_ibov"])
    parser.add_argument("--otimizar-pesos", action="store_true",
                        help="após escolher os ativos, otimizar os pesos (senão pesos iguais)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"1/4 Obtendo universo {args.indice.upper()} e validando dados "
          f"({args.inicio} a {args.fim})...")
    tickers = universo.obter_universo(args.indice, online=not args.offline)
    precos = dados.baixar_precos(tickers, args.inicio, args.fim)
    print(f"    {len(precos.columns)}/{len(tickers)} ativos com dados no período.")

    retornos = dados.calcular_retornos(precos)
    benchmark = dados.calcular_retornos(dados.baixar_benchmark(args.inicio, args.fim))

    print(f"2/4 Buscando melhor carteira em {args.num_carteiras} combinações "
          f"(critério: {args.criterio})...")
    selecao = otimizacao.buscar_carteira_aleatoria(
        retornos,
        num_carteiras=args.num_carteiras,
        num_ativos=args.num_ativos,
        criterio=args.criterio,
        retornos_benchmark=benchmark,
        seed=args.seed,
    )
    if args.otimizar_pesos:
        print("    Otimizando pesos da carteira selecionada (SLSQP)...")
        selecao = otimizacao.otimizar_pesos(
            retornos, selecao.carteira.tickers, args.criterio, benchmark
        )
    cart = selecao.carteira

    print(f"3/4 Simulando {args.num_simulacoes} trajetórias de {args.horizonte} dias úteis "
          "(Monte Carlo multivariado)...")
    simulacao = SimulacaoMonteCarlo(precos, cart, seed=args.seed)
    simulacao.simular(num_dias=args.horizonte, num_simulacoes=args.num_simulacoes)
    estatisticas = simulacao.estatisticas_finais()

    print("4/4 Relatório\n")
    print("=" * 70)
    print("CARTEIRA SELECIONADA (in-sample — validar com walk-forward!)")
    print("=" * 70)
    for ticker, peso in zip(cart.tickers, cart.pesos):
        print(f"  {ticker:<12} peso {peso:6.2%}")
    print(f"\nCritério de seleção: {selecao.criterio} = {selecao.valor_criterio:.4f}")
    print("\nMétricas da janela de seleção:")
    for nome, valor in selecao.metricas.items():
        print(f"  {nome:<26} {valor:10.4f}")
    print(f"\nSimulação de Monte Carlo ({args.horizonte} dias úteis à frente):")
    print(f"  Probabilidade de terminar acima do valor atual: {estatisticas['prob_acima_atual']:.1%}")
    print(f"  Retorno esperado (média):  {estatisticas['media']:.4f}")
    print(f"  Cenário pessimista (p5):   {estatisticas['p05']:.4f}")
    print(f"  Mediana (p50):             {estatisticas['mediana']:.4f}")
    print(f"  Cenário otimista (p95):    {estatisticas['p95']:.4f}")

    saida = RAIZ / "Resultados" / "pipeline_ultima_execucao.json"
    saida.write_text(json.dumps({
        "janela": {"inicio": args.inicio, "fim": args.fim},
        "carteira": dict(zip(cart.tickers, cart.pesos.round(4).tolist())),
        "criterio": {"nome": selecao.criterio, "valor": selecao.valor_criterio},
        "metricas_selecao": selecao.metricas,
        "monte_carlo": estatisticas,
        "parametros": {
            "num_carteiras": args.num_carteiras,
            "num_simulacoes": args.num_simulacoes,
            "horizonte_dias": args.horizonte,
            "seed": args.seed,
        },
    }, indent=2, ensure_ascii=False))
    print(f"\nResultado salvo em {saida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
