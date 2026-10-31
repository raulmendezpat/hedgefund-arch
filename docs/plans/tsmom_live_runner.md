# Ejecutor en vivo de la TSMOM BTC/ETH (infraestructura, no decisión)

Escrito el 2026-10-31. Prepara la infraestructura para que, **si** la TSMOM o la combinación
aprueban el 2027-04-19 (`tsmom_paper_plan.md`), pueda arrancar sin construir nada a última hora.
**No cambia el plan registrado:** la TSMOM no opera en vivo antes de esa decisión, y la decisión
no depende de este ejecutor.

## Qué es

`scripts/live_tsmom.py` (wrapper `deploy/run_live_tsmom.sh`):
- **Señal, calendario y banda:** iguales a los del paper. Usa la misma función de objetivos que
  `scripts/paper_tsmom.py`, decide cada día *d* y ejecuta en *d*+1 a la 01:00 UTC. Un fill con más
  de 12 h de atraso se registra como perdido.
- **Controles:** los mismos que el ejecutor de la XS (`hf_research.live_xs`):
  - kill switch con −35 % de caída desde el máximo: cierra todo y pausa hasta `reset-risk`;
  - −8 % en el día: modo solo-reducir;
  - límites de cordura que abortan el rebalanceo;
  - las transferencias no cuentan como P&L;
  - aborta ante posiciones que no abrió él.
- **Modos:** `dry-run` (por defecto, no envía nada) y `live`, que además exige `LIVE_TRADING=1`.

**Diferencias con la configuración de la XS:**

| Parámetro | XS | TSMOM | Motivo |
|---|---|---|---|
| Peso máximo por activo | 0.10 | 1.0 | `TsmomConfig.max_weight` |
| Exposición bruta máxima | 1.0 | 2.0 | dos activos de hasta 1.0 cada uno |
| Atraso máximo del fill | 24 h | 12 h | igual que el paper |
| Desvío post-trade reportado | 1 % | 3 % | ETH se opera en pasos de 0.01 (~22 USDT) |
| Alerta de pérdida por posición | 1.5 % | 3 % | posiciones de ~20 % del capital cada una |

## Subcuenta propia (obligatoria)

La XS también opera BTC y ETH, y cada ejecutor aborta ante las posiciones del otro. Por eso la TSMOM:
- lee credenciales **solo** de `~/.config/hedgefund/bitget_tsmom.env` (o `$BITGET_TSMOM_ENV_FILE`),
  con chmod 600 y una key sin permiso de retiro;
- nunca usa el archivo de la XS ni las variables `BITGET_*`;
- rechaza una key igual a la de la XS.

## Capital mínimo

El mínimo de ETH en Bitget es 0.01 (~22 USDT). Con el día de prueba 2026-10-29 (BTC +0.186,
ETH −0.162), la orden de ETH queda redondeada a:

| Capital | ETH objetivo | ETH enviado | Error de redondeo |
|---|---|---|---|
| 300 | 48.6 USDT | 0.02 (44.9) | −8 % |
| 500 | 81.0 USDT | 0.03 (67.4) | −17 % |
| 1,000 | 162 USDT | 0.07 (157) | −3 % |
| 2,000 | 324 USDT | 0.14 (315) | −3 % |

Además, los ajustes diarios que pasan la banda son del 5 % del capital o más, y deben superar
~22 USDT en ETH. Con menos de ~500 USDT muchos se saltan, y el libro se aparta del paper.
**Capital recomendado: ≥ 1,000 USDT** en la subcuenta de la TSMOM.

## Pasos para activarlo (cuando el dueño de la cuenta lo decida)

1. Crear una subcuenta nueva en Bitget y una API key solo de trading, sin retiro. Guardarla en
   `~/.config/hedgefund/bitget_tsmom.env` con chmod 600.
2. Dry-run: `init --mode dry-run`, agente de launchd en el minuto 17 con `MODE=dry-run` y el
   runner en el healthcheck (`--runner tsmom-dry-run=<state>:<logs>`). Correrlo unas semanas
   y verificar que sus objetivos coincidan con el paper todos los días.
3. Vivo, solo si la decisión del 2027-04-19 lo aprueba y el dueño lo confirma:
   - `init --mode live`;
   - agente con `MODE=live LIVE_TRADING=1`;
   - capital según el plan de escalado de la XS;
   - VPS si el total supera 1,000 USDT.
