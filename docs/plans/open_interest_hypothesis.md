# Hipótesis registrada: señales de open interest (apalancamiento acumulado)

Escrita el 2026-10-21, **antes de descargar los datos de open interest y de mirar ningún
resultado**. Solo la señal primaria decide; las secundarias son exploratorias. La configuración y
los criterios no se cambian después de ver los resultados.

## Datos

Archivos públicos de Binance USDT-M: `data.binance.vision/data/futures/um/daily/metrics/`. Hay un
archivo por símbolo y día, con filas cada 5 minutos:
- `sum_open_interest`: contratos abiertos, en unidades del activo;
- `sum_open_interest_value`: open interest en USDT;
- `count_long_short_ratio`: cuentas en largo / cuentas en corto;
- `sum_toptrader_long_short_ratio`: lo mismo, solo para los top traders;
- `sum_taker_long_short_vol_ratio`: volumen de compradores agresivos / vendedores agresivos.

**Valor diario del día *d*:** la última fila del día UTC con marca ≤ 23:55. Se conoce antes del
cierre de la señal (24:00) y del fill (01:00).

**Universo:** se descargan solo los símbolos del top 60 mensual
(`artifacts/research/universe_top60_v1`), desde un mes antes de su entrada. Un símbolo sin dato
ese día no es elegible.

**Cobertura:**
- Se informa la fracción del universo con dato cada mes.
- El período de desarrollo empieza el primer mes con cobertura ≥ 80 %, pero nunca antes del
  2021-03-01.

## Señal primaria: reversión del apalancamiento acumulado (cruzada)

**Idea:**
- Un crecimiento fuerte de contratos abiertos indica posiciones apalancadas nuevas que se
  acumulan. Esas posiciones tienden a deshacerse (liquidaciones y cierres forzados), y eso
  presiona el precio en contra de la dirección acumulada.
- Medir en contratos, y no en USDT, separa el apalancamiento nuevo del simple movimiento del
  precio.

**Score:** score_i(d) = ln(OI_i(d) / OI_i(d−7)), con OI = `sum_open_interest` (contratos).

**Cartera:**
- **Dirección pre-registrada:** contraria. Se **vende en corto el 20 % con mayor** crecimiento de
  contratos y se **compra el 20 % con menor** crecimiento (desapalancamiento).
- Construcción igual al momentum primario:
  - pesos iguales, 0.5 largo / 0.5 corto;
  - rebalanceo semanal con banda de 0.02;
  - mínimo 8 activos con score.
- Costos escalonados (3/6/12 bp + fee de 0.06 %) y funding real.
- Decisión a las 23:00 UTC del día *d* y fill a la 01:00 del día *d*+1, con el mismo motor
  `hf_research.engine`.

**Períodos:** desarrollo desde el inicio definido arriba hasta 2023-12-31; fuera de muestra desde
2024-01-01.

### Criterios de éxito (todos requeridos)

| # | Criterio |
|---|---|
| 1 | Sharpe fuera de muestra ≥ 0.7 |
| 2 | Sharpe de desarrollo ≥ 0.4 |
| 3 | Peor caída fuera de muestra ≥ −35 % |
| 4 | Sharpe fuera de muestra sin los 5 símbolos que más aportaron ≥ 0.4 |
| 5 | Bootstrap por bloques de 20 días: probabilidad de Sharpe fuera de muestra ≤ 0 menor a 10 % |

**Combinación con el momentum primario:**
- Riesgo igual, con la escala calculada solo con datos de desarrollo.
- Es útil si, fuera de muestra, Sharpe ≥ momentum + 0.15 **y** la peor caída es mejor.
- Se informa la correlación diaria con el momentum.

Si la dirección contraria falla y la dirección opuesta (seguir el crecimiento de OI) "funciona",
eso **no** cuenta como hallazgo: requiere una pre-registración nueva y datos nuevos.

## Señales secundarias (exploratorias, no deciden nada)

Se construyen igual que la primaria y se informan con la advertencia de que son 3 pruebas
adicionales: alguna puede salir bien por azar.

| Señal | Score | Dirección |
|---|---|---|
| S1. Posicionamiento minorista | `count_long_short_ratio` del día *d* | corto al 20 % más largo, compra al 20 % más corto |
| S2. Flujo agresivo | promedio de 7 días de ln(`sum_taker_long_short_vol_ratio`) | compra al 20 % más comprador, corto al 20 % más vendedor |
| S3. OI que confirma el precio | score primario × signo del retorno de 7 días | corto donde el apalancamiento crece a favor del movimiento |

Una secundaria con Sharpe fuera de muestra ≥ 0.7 solo justifica una pre-registración nueva, que
tendría que validarse en paper.

## Sensibilidad de la primaria (solo robustez)

- Ventana: 3, 7 y 14 días.
- Fracción por lado: 10 %, 20 % y 30 %.
- Rebalanceo: semanal y cada 3 días.
- Costos ×2.

Se informa la fracción de las 18 combinaciones con Sharpe fuera de muestra ≥ 0.5.

## Limitaciones conocidas

- **Fuente:** el open interest es de Binance y la ejecución sería en Bitget. El posicionamiento
  de Bitget puede diferir.
- **Calidad de los datos:** Binance publica estos archivos con huecos ocasionales. Un hueco deja
  al símbolo fuera ese día.
- **Herencia del momentum:** se hereda el universo y la construcción del momentum, así que los
  resultados pueden estar correlacionados con él. Por eso se informa la correlación y la
  combinación.

## Resultado (2026-10-21): primaria rechazada; una secundaria (flujo agresivo) amerita pre-registración nueva

Corrida: `scripts/run_open_interest.py` → `artifacts/research/open_interest_v1/`. Datos:
`scripts/fetch_binance_metrics.py`, 144,638 días-símbolo, 12.7 % sin dato.

**Cobertura:**
- Antes de 2021-12 el archivo solo tiene datos de ~2 % del universo.
- Por la regla del 80 %, el desarrollo empieza el **2021-12-01**: son 2 años.

### Primaria: reversión del apalancamiento acumulado (rechazada en todo)

| Criterio | Umbral | Resultado | ¿Cumple? |
|---|---|---|---|
| Sharpe fuera de muestra | ≥ 0.7 | −0.76 | no |
| Sharpe de desarrollo | ≥ 0.4 | −0.67 | no |
| Peor caída fuera de muestra | ≥ −35 % | −75 % | no |
| Sharpe sin los 5 mejores símbolos | ≥ 0.4 | −1.43 | no |
| Bootstrap P(Sharpe ≤ 0) | < 10 % | 86 % | no |

- **No hay señal en ninguna dirección:**
  - La dirección opuesta, informada solo como control, también pierde: −0.81 en desarrollo y
    −0.22 fuera de muestra.
  - Con rotación de 88 veces el capital al año, los costos (~13 %/año) se comen todo.
- **Grilla:** 0 de 18 combinaciones con Sharpe ≥ 0.5 y mediana −0.88. Con rebalanceo cada
  3 días y un 10 % por lado, la cuenta se pierde entera.
- **Combinación con momentum:** empeora. Sharpe 0.03 contra 1.00.

### Secundarias (exploratorias: 3 pruebas extra, no deciden nada)

| Señal | Sharpe desarrollo | Sharpe fuera de muestra | Peor caída fuera | Precio / funding / costos fuera (%/año) | Correlación con momentum (fuera) |
|---|---|---|---|---|---|
| S1 posicionamiento minorista (contraria) | 1.87 | 0.73 | −39 % | +8.7 / +16.8 / −5.4 | 0.38 |
| S2 flujo agresivo (seguidora) | 1.28 | 0.93 | −25 % | +25.4 / +8.9 / −9.8 | 0.38 |
| S3 OI × signo del precio | −0.51 | −1.18 | −76 % | −18.9 / −3.2 / −11.9 | — |

- **S2** (comprar las monedas con más presión compradora agresiva en 7 días y vender en corto
  las de más presión vendedora):
  - Es la única con retorno de precio consistente: +27 %/año en desarrollo y +25 %/año fuera de
    muestra.
  - Sharpe por año: 2022 1.51 · 2023 1.13 · 2024 −0.17 · 2025 0.66 · 2026 2.00.
  - Combinada con momentum a la misma volatilidad (exploratorio): Sharpe 0.87 contra 0.22 en
    desarrollo, y 1.19 contra 1.03 fuera de muestra.
- **S1:** fuera de muestra vive casi solo del funding (+16.8 %/año). En la práctica es un carry
  disfrazado. Además, 2024 tuvo Sharpe −1.39.
- **S3:** igual de mala que la primaria.

**Conclusión:**
- La reversión del apalancamiento no existe en estos datos.
- S2 cumple la regla de esta pre-registración (Sharpe fuera de muestra ≥ 0.7), pero salió de entre
  4 pruebas y todo el historial disponible ya fue mirado.
- La única validación honesta que queda es **un paper con una pre-registración nueva**, porque no
  hay datos históricos sin mirar.
