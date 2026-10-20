# Plan de paso a producción — momentum cross-sectional en Bitget

Escrito el 2026-10-20, **antes del primer rebalanceo del paper** (2026-10-26). Las reglas se
fijan ahora para que los resultados no las influyan. Cambiarlas después requiere dejar
constancia aquí del motivo, con fecha, **antes** de mirar los datos que motivarían el cambio.

Las decisiones marcadas como *propuesta, por confirmar* son las opciones por defecto;
falta la confirmación del dueño de la cuenta.

## 0. Qué se pone en producción

La hipótesis registrada, sin cambios:
- **Señal y cartera:** largo/corto, retorno de 30 días con 1 día de salto; 20% de los
  símbolos por lado, pesos iguales.
- **Calendario:** rebalanceo semanal, con decisión al cierre de las 23:00 UTC del domingo y
  ejecución el lunes a las 01:00 UTC.
- **Banda:** 0.02.
- **Universo:** top 60 de Binance por volumen, listados en Bitget.
- **Exposición:** 1x bruto, apalancamiento de Bitget 3x, margen cross.
- **Código:** `scripts/live_xs_momentum.py` en modo `live`, en un commit fijado.

**Lo que se sabe de la estrategia** (pruebas de estrés):
- Sharpe fuera de muestra de 1.08, que baja a 0.86–1.01 con costos realistas.
- El intervalo del bootstrap al 5–95% va de 0.28 a 1.99.
- 5 tokens nuevos explicaron el 79% del retorno fuera de muestra.

Es decir: hay ventaja, pero es frágil y está concentrada. El plan está pensado para eso:
capital chico al inicio, escalado lento y criterios de parada claros.

## 1. Puerta de entrada (fin del paper)

Revisión tras el rebalanceo n.º 12 del paper (ejecución del **2027-01-11**). Hay que cumplir
**todos** los puntos:

| Criterio | Umbral | Fuente |
|---|---|---|
| Rebalanceos ejecutados | ≥ 12 | `paper_xs_momentum.py report` |
| Rebalanceos perdidos | ≤ 1 | idem |
| Slippage mediano por tier (top 10 / 11–30 / 31–60) | ≤ 3 / 6 / 12 bp | idem |
| Correlación diaria paper vs modelo | ≥ 0.9 | idem |
| Brecha de implementación paper vs modelo | ≥ −5 %/año | idem |
| Dry-run: rebalanceos con órdenes generadas sin error | ≥ 4 | `live_xs_momentum.py status` |
| Dry-run: aborts, sanity aborts o errores Python sin explicar | 0 | monitor + logs |

**Qué mide y qué no:**
- 12 semanas **no** permiten juzgar la rentabilidad. Con un Sharpe de 1, el error estándar
  del retorno en 12 semanas es mayor que el retorno esperado.
- La puerta mide si **la implementación reproduce el modelo**, no si gana dinero.
- El riesgo de que la estrategia no funcione se controla con el tamaño del capital (§2–3)
  y con los criterios de parada (§4).

**Si falla un criterio:** no se pasa a producción. Se documenta la causa. Si es corregible
(por ejemplo, un problema de ejecución), se corrige y se repiten 12 semanas de paper.

## 2. Arranque

- **Capital inicial: 1,000 USDT** *(propuesta, por confirmar; alternativas: 300 o 3,000)*.
  - Son unos 42 USDT por pata, por encima del mínimo de Bitget (~5 USDT) y con poco error
    de redondeo.
  - Pérdida máxima antes del kill switch: ~250 USDT.
- **Servidor: VPS con IP fija** *(propuesta, por confirmar; alternativa: este Mac)*.
  - La API key de la subcuenta se amarra a esa IP.
  - Permisos de la key: solo futuros, lectura y operación; nunca retiros.
- **Pasos** (primera semana tras aprobar la puerta):
  1. Instalar el VPS con el commit fijado, el monitor y el keep-alive del servicio.
  2. Correr el dry-run en el VPS una semana en paralelo con el Mac, hasta que ambos
     generen las mismas órdenes.
  3. Transferir el capital a la subcuenta. La transferencia se detecta sola; no hace falta
     ajustar nada.
  4. `init --mode live` y activar el job con `MODE=live LIVE_TRADING=1`.
  5. Tras el primer rebalanceo: `report --mode live --paper-dir …` y revisión manual de
     cada orden.
- El paper **sigue corriendo** en paralelo como referencia mientras exista la estrategia en vivo.

## 3. Escalado

**Por etapas** *(propuesta, por confirmar; alternativas: fijo 12 semanas o decidir después)*.

Cada 4 rebalanceos en vivo se revisa:

| Chequeo | Umbral |
|---|---|
| Slippage + comisión reales, mediana por tier | ≤ modelo (3/6/12 bp de slippage) |
| Correlación diaria vivo vs paper | ≥ 0.9 |
| Rebalanceos perdidos o abortados | 0 |
| Estado de riesgo | ni `halted` ni `reduce_only` en el período |

- **Todos OK:** se duplica el capital (1k → 2k → 4k → 8k).
- **Falla uno:** se mantiene el capital.
- **Dos revisiones seguidas con fallas:** se baja una etapa.
- **Tope inicial: 10,000 USDT** *(propuesta, por confirmar)*. Subirlo exige repetir
  `check_bitget_universe.py` al tamaño de orden nuevo y que el costo del libro siga dentro
  del modelo.

El escalado **no** depende del retorno obtenido: cuatro semanas buenas o malas son ruido.

## 4. Criterios de parada

**Automáticos** (ya implementados en el ejecutor):
- **Drawdown ≥ 25 % desde el máximo:** kill switch. Cierra todo y se detiene.
  `reset-risk` solo después de la revisión de §4.3; nunca el mismo día.
- **Pérdida diaria ≥ 8 %:** modo solo-reducir hasta el día UTC siguiente.
- **Límites de sanidad:** peso > 10 %, bruto > 1x, orden > 5,000 USDT o precio faltante.
  Se aborta el rebalanceo.
- **Posiciones ajenas o en modo hedge:** se aborta.

**De estrategia** (revisión manual, se evalúan en cada revisión de 4 semanas):
- **Retorno acumulado en vivo < −15 %:** se para el escalado y se vuelve a la etapa inicial.
- **Tras 26 semanas en vivo, Sharpe en vivo < 0 y además por debajo del paper en el mismo
  período:** se detiene la estrategia y se revisa la hipótesis.
- **Brecha vivo vs paper peor que −5 %/año sostenida en 2 revisiones:** se detiene y se
  investiga la ejecución.

**Decisión abierta: umbral del kill switch** *(por decidir)*.
- El backtest x1 con marcas horarias (`docs/plans/vol_targeting_hypothesis.md`) habría tocado
  −25 % en 2021-08-26 y en 2024-08-28. Su peor caída horaria fue −26 % en desarrollo y −35 %
  fuera de muestra.
- Con −25 %, una caída normal para esta estrategia la detiene.
- Opciones:
  - subir el umbral a ~−40 %;
  - mantener −25 % sabiendo que es probable que se active;
  - mantenerlo como pausa de revisión y no como cierre definitivo, que es lo que ya dice el
    punto siguiente.

**Revisión tras un kill switch:**
- **Causa de mercado** (la estrategia simplemente perdió): se reinicia en la etapa inicial
  solo si el backtest muestra drawdowns comparables.
- **Causa de implementación:** se corrige, pasa por CI y se hace dry-run antes de reiniciar.

## 5. Riesgo de cola del lado corto

Ejemplo observado: BR subió +47 % en un día mientras estaba en corto en la simulación del
19-oct; costó ~2 % de la cartera con un peso de solo 4.2 %.

**Decisión: aceptar y vigilar** *(propuesta, por confirmar)*. Motivos:
- No se cambia la hipótesis probada.
- Las guardas automáticas (−8 % diario, −25 % total) acotan la pérdida de la cartera.
- **Alerta por posición** (implementada el 2026-10-20): el monitor avisa una vez por
  posición y rebalanceo cuando una sola posición pierde más de 1.5 % del equity desde el
  último rebalanceo, tanto en el paper como en vivo. Solo avisa; no opera.

Las variantes (un stop por posición, o no shortear tokens con menos de 90 días listados)
son **hipótesis nuevas**: requieren su propio backtest y su propio paper antes de entrar a
producción. No se aplican a mitad de la prueba.

## 6. Operación

- **Cada lunes** (~01:20 UTC): el monitor avisa del rebalanceo. Se revisan la notificación,
  `status` y, en vivo, `report`.
- **Cada 4 semanas:** revisión de escalado (§3) y de criterios de estrategia (§4).
- **Cada 90 días:** rotar la API key de la subcuenta.
- **Depósitos y retiros:** solo por transferencia interna de Bitget; el ejecutor los detecta.
- **Cambios de código:** rama → CI verde → dry-run en el servidor → mover el commit fijado.
  Nunca editar archivos en el worktree o servidor en producción.

## 7. Fechas

| Fecha | Hito |
|---|---|
| 2026-10-26 | Primer rebalanceo del paper y del dry-run |
| 2026-11-23 | Revisión intermedia del paper (5 rebalanceos): solo funcionamiento, sin decisiones |
| 2027-01-11 | Rebalanceo n.º 12: revisión de la puerta (§1) |
| ~2027-01-18 | Arranque en vivo si pasa la puerta y el VPS está listo |
| ~2027-02-15 | Primera revisión de escalado (4 rebalanceos en vivo) |
