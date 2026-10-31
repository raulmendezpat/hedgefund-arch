# Hipótesis registrada: momentum cruzado con 10 % por lado, validada solo con datos nuevos

Escrita el 2026-10-24, antes del primer rebalanceo de su paper (2026-10-26). Como taker-flow, esta
variante **ya se vio en todo el historial**: salió de un diagnóstico de la selección del momentum
hecho sobre los mismos datos del backtest. El backtest no puede validarla. La validación es
**exclusivamente el paper**, con datos posteriores a esta fecha y la configuración congelada aquí.

## De dónde sale

Diagnóstico de la selección de la hipótesis principal (`xs_momentum_production_plan.md`), semana a
semana desde 2021-03, con el P&L a 7 días de cada token neto de funding y relativo al promedio del
universo:
- La ventaja se concentra en los extremos del ranking: el 5 % superior y el 5 % inferior. El tramo
  del 5 al 20 % de cada lado aporta cerca de cero.
- Lo que queda justo fuera del corte (percentiles 20–30 y 70–80) rinde como el centro del ranking:
  el corte no deja fuera señales buenas.
- Los puestos 61–120 por liquidez rinden peor que el top 60: ampliar el universo no ayuda.

Backtest con el motor compartido y costos del backtest (fee 0.06 %, slippage 3 bp), top 60:

| | Sharpe desarrollo (2021-03 a 2023) | Sharpe fuera de muestra (2024+) | Peor caída desarrollo / fuera | Costo anual |
|---|---|---|---|---|
| Principal (20 % por lado) | 0.42 | 1.07 | −24 % / −32 % | 4.0 % |
| **Variante (10 % por lado)** | **0.51** | **1.38** | **−39 % / −32 %** | 4.7 % |

- Por año, la variante gana en 2021, 2024 y 2026 y pierde en 2022, 2023 y 2025. La mejora viene de
  pocos años buenos.
- Correlación diaria con la principal: 0.82–0.87.
- La caída de −39 % en desarrollo supera el kill switch de producción (−35 %).

**Lo que se sacrifica:** con ~55–60 símbolos operables quedan ~6 por lado y un peso de ~8.3 % por
posición, el doble que la principal. El riesgo de cola por token (ver §5 del plan de producción)
pesa el doble.

## Configuración (congelada)

Idéntica a la hipótesis principal salvo la fracción por lado:
- **Señal:** retorno de 30 días con 1 día de salto, sobre cierres de 1h de Binance a las 23:00 UTC.
- **Universo:** top 60 mensual de Binance USDT-M por volumen, restringido a símbolos listados en Bitget.
- **Cartera:** largo el **10 %** con mayor score y corto el **10 %** con menor score, mínimo 2 por
  lado; pesos iguales 0.5 / −0.5; mínimo 8 activos con score; banda de 0.02.
- **Calendario y ejecución:** idénticos al paper del momentum. Decisión el domingo, fill el lunes
  a la 01:00 UTC, simulación contra el libro de Bitget, fee de 0.06 % y funding real de Bitget.
- **Capital simulado:** 10,000 USDT.

## Compuerta de implementación (2027-01-11, 12 rebalanceos)

Mismos umbrales que el paper del momentum:
- ≥ 12 rebalanceos y ≤ 1 perdido;
- slippage mediano por tramo ≤ 3 / 6 / 12 bp;
- correlación diaria paper vs modelo ≥ 0.9;
- brecha de implementación ≥ −5 %/año.

## Decisión (solo con datos desde el primer rebalanceo, 2026-10-26)

Se compara contra el paper de la principal (`xs_momentum_v1`) en los mismos días, porque la
pregunta es si la variante es **mejor que la principal**, no solo si gana dinero.

| Fecha | Regla |
|---|---|
| 2027-04-19 (26 semanas) | Se **descarta** si su peor caída es < −35 %, o si su Sharpe anualizado es < −0.5. Si no, sigue corriendo. |
| 2027-10-25 (52 semanas) | **Reemplaza a la principal** en producción solo si cumple todo: Sharpe ≥ 0.7, Sharpe ≥ el de la principal + 0.3 en los mismos días, peor caída ≥ −35 % y compuerta de implementación aprobada. Si no, se descarta y la principal sigue igual. |

**Requisito antes de producción:** el límite de sanidad del ejecutor aborta con pesos > 10 %. Con
menos de 50 símbolos operables quedan 5 por lado (10 % exacto), y con menos de 45, 4 por lado
(12.5 %), que abortaría el rebalanceo. Hay que revisar ese límite antes de pasarla a vivo.

**Advertencia de potencia:** la diferencia diaria entre ambas tiene un Sharpe histórico de 0.4–1.1.
En 52 semanas eso da un estadístico t de 0.4–1.1: aun si la mejora fuera real, un año de paper
probablemente no la distinga del ruido. La regla exige una mejora grande a propósito, porque la
variante duplica la concentración y un falso positivo cuesta capital real.

## Variante descartada antes del paper: largos solo en tokens con menos de un año listados

Salió del mismo diagnóstico. Los largos en tokens listados hace menos de 365 días (según
`onboardDate` de Binance) rindieron más que los viejos: +79 bp/semana contra +1 en desarrollo y
+401 contra −65 fuera de muestra. Como cartera (cortos iguales a la principal, largos solo en
tokens jóvenes del 20 % superior), quedó **peor que la principal en desarrollo**: Sharpe 0.31
contra 0.49, con caída de −42 % contra −18 %, sin costos. El motivo es que hay muy pocos: una
mediana de 3 largos por semana, y menos de 2 en 54 de 289 semanas.

Decisión del 2026-10-24: **no se registra**. Queda anotada para no volver a probarla sin una razón
nueva.
