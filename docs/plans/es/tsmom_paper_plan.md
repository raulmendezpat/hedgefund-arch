# Plan registrado: paper de TSMOM BTC/ETH y su combinación con el momentum cruzado

Escrito el 2026-10-21, antes de arrancar el paper. Nada de lo que sigue se cambia mientras
corre.

## Por qué un paper de una hipótesis que no pasó

La TSMOM no cumplió sus criterios pre-registrados (`tsmom_btc_eth_hypothesis.md`): Sharpe fuera de
muestra 0.63 contra 0.7, y caída −33.7 % contra −30 %. Aun así, es la estrategia más robusta
probada: positiva todos los años, en las 24 variantes y con costos ×2. No se bajan los umbrales.
En cambio, se juntan **datos nuevos** con la configuración primaria intacta y luego se vuelven a
aplicar **los mismos criterios** sobre un período fuera de muestra más largo.

## Qué corre

**Libro TSMOM** (`runtime/paper/tsmom_btc_eth_v1`):
- Capital simulado: 10,000 USDT.
- Configuración: `TsmomConfig()` primaria, sin cambios.
  - Señal 20/60/120 días sobre cierres de Binance a las 23:00 UTC.
  - Volatilidad objetivo de 20 % por activo, tope de peso 1.0.
  - Banda de 0.05, modo largo/corto.
- Ejecución simulada contra el libro de órdenes de Bitget (BTC/USDT y ETH/USDT), con fee de
  0.06 %. Se aplica el funding real de Bitget.
- Rebalanceo diario: decisión el día *d*, fill el día *d*+1 a la 01:00 UTC. Un fill con más de
  12 h de atraso se registra como perdido.
- Nunca envía órdenes: solo usa datos públicos.

**Combinación:**
- Se evalúa a partir de los dos papers, sin un tercer libro: retorno diario = 0.5 × momentum
  cruzado (`xs_momentum_v1`) + 0.5 × TSMOM.
- La escala de la TSMOM es 1.0: en el backtest dio 0.98 para igualar la volatilidad de desarrollo.
- Se compara con el momentum cruzado solo, en los mismos días.

## Compuerta de implementación (2027-01-11, junto con la del momentum cruzado)

| Chequeo | Umbral |
|---|---|
| Rebalanceos diarios hechos | ≥ 70 |
| Rebalanceos perdidos | ≤ 3 |
| Slippage mediano (BTC/ETH, tramo 1) | ≤ 3 bp |
| Correlación diaria paper vs modelo (replay con datos de Binance) | ≥ 0.9 |
| Brecha de implementación | ≥ −5 %/año |

Doce semanas no alcanzan para juzgar rentabilidad. Esta compuerta solo valida que el paper
reproduce el modelo.

## Decisión sobre la combinación (2027-04-19, tras 26 semanas)

1. Se repite `scripts/run_tsmom.py` sin cambios en el código ni en los criterios. El período
   fuera de muestra se extiende de 2024-01-01 a la fecha del corte.
2. La combinación pasa a producción (con el capital y las escalas del plan del momentum cruzado)
   solo si se cumplen las dos condiciones:
   - la compuerta de implementación de la TSMOM está aprobada;
   - con el período extendido, **o** la TSMOM primaria cumple sus 4 criterios **o** la
     combinación cumple los suyos: Sharpe ≥ momentum + 0.15, caída mejor y exposición
     implementable.
3. Si no, la TSMOM se descarta y queda solo el momentum cruzado.

**Advertencia:** volver a evaluar los mismos criterios con más datos es una prueba secuencial.
Suma una segunda oportunidad de aprobar. Se acepta porque los criterios y el código no cambian,
y porque la mitad de los datos nuevos (la del paper) es genuinamente fuera de muestra.
