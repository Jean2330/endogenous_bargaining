# Overview: timing y reglas del juego

Documento de trabajo para la reunión con Itza. Resume el juego tal como está hoy en `bargainigpower_draft.tex` (parte estática), en la parte dinámica de `thesis_draft.tex` y en la implementación numérica (`src/`). La sección 8 lista los huecos e inconsistencias que hay que decidir antes de reescribir el modelo.

## 1. Piezas del juego

- **Jugadores**: principal neutral al riesgo (él) y agente con utilidad CARA (ella).
- **Tiempo**: discreto, $t = 0, 1, 2, \dots$, horizonte infinito, factor de descuento común $\rho$.
- **Estado**: la riqueza acumulada del agente, $W_t$. Es información pública.
- **Contrato**: lineal, $w_t = \alpha_t + \beta_t y_t$. Dura un solo periodo.
- **Tecnología**: $y_t = \theta a_t + \sigma \varepsilon_t$, $\varepsilon_t \sim N(0,1)$ i.i.d.; costo de esfuerzo $\frac{k}{2}a_t^2$.
- **Restricción de responsabilidad limitada (LL)**: $\alpha_t + \beta_t \underline{y} \ge \underline{w}(W_t) = \overline{w} - \gamma_w W_t$. Mientras más rica la agente, más laxa la restricción.
- **Reglas de reparto**: Nash bargaining asimétrico con peso exógeno $\delta$ para la agente y pagos de desacuerdo exógenos $(\overline{\Pi}, \overline{CE})$.

## 2. Timing dentro de un periodo $t$

| Paso | Qué ocurre | Quién sabe qué |
|---|---|---|
| 0 | Se observa $W_t$. Con él queda fijo el piso $\underline{w}(W_t)$ y, por lo tanto, el conjunto de contratos factibles $\Gamma(W_t)$. | $W_t$, los parámetros $(\theta, k, \gamma, \sigma, \delta, \rho, \overline{w}, \gamma_w, \underline{y})$ y los pagos de desacuerdo son conocimiento común. |
| 1 | Negociación Nash sobre el par $(\alpha_t, \beta_t)$, conjuntamente. | Ninguno conoce todavía $\varepsilon_t$. El principal no observa el esfuerzo. |
| 2 | La agente elige $a_t$ dado el contrato firmado. Por IC, $a_t^* = \beta_t \theta / k$. | La agente sabe $\beta_t$ al elegir. El principal anticipa $a_t^*$ porque la IC lo determina. |
| 3 | Se realiza $\varepsilon_t$ y por tanto $y_t$. | $y_t$ es público (contratable). $a_t$ no se observa, aunque ambos lo infieren desde $\beta_t$. |
| 4 | Se paga $w_t = \alpha_t + \beta_t y_t$. | Público. |
| 5 | Se actualiza la riqueza: $W_{t+1} = W_t + w_t - \frac{k}{2}a_t^2$. | Público. Empieza el periodo $t+1$ con contrato nuevo. |

Puntos del timing que sostienen el modelo:

- El contrato se firma antes del esfuerzo y no se renegocia dentro del periodo. Si se pudiera renegociar con el esfuerzo ya hundido, las partes volverían a $\beta = 0$ y el incentivo no tendría efecto (Remark sobre renegociación en el draft).
- Entre periodos la renegociación es total. No hay compromiso intertemporal: cada periodo se firma un contrato nuevo, y lo único que lleva de un periodo al siguiente es $W$.

## 3. Cómo se determinan $\alpha_t$ y $\beta_t$

Ambos salen de resolver, en el paso 1,

$$\max_{(\alpha,\beta)\in\Gamma(W_t)} \; [\Pi - \overline{\Pi}]^{1-\delta}\,[CE - \overline{CE}]^{\delta},$$

con $\Pi = D(\beta) - \alpha$, $CE = \alpha + C(\beta)$, $D(\beta) = \frac{\theta^2}{k}\beta(1-\beta)$ y $C(\beta) = \frac{\beta^2}{2}\left(\frac{\theta^2}{k} - \gamma\sigma^2\right)$. La IC ya está sustituida. La LL es la única restricción que muerde, y de si está activa o no depende cómo se reparte el trabajo entre $\alpha$ y $\beta$.

Sea $\nu = \theta^2/k$ y $\kappa = \gamma\sigma^2$.

**Régimen holgado ($W_t \ge \overline{W}$).**
- $\alpha$ es una transferencia pura (suma 1 al agente y resta 1 al principal), así que la frontera de Pareto es una recta de pendiente $-1$.
- $\beta$ se fija en el nivel que maximiza el excedente, $\beta^{\text{eff}} = \nu/(\kappa + \nu)$.
- $\alpha$ reparte: $\alpha^* = \overline{CE} - C(\beta^{\text{eff}}) + \delta[S^{\text{eff}} - \overline{CE} - \overline{\Pi}]$.
- La agente se queda exactamente con la fracción $\delta$ del excedente neto.

**Régimen restringido ($W_t < \overline{W}$).**
- La LL ata $\alpha_t = \underline{w}(W_t) - \beta_t \underline{y}$, así que $\alpha$ deja de ser libre.
- Solo queda $\beta$ para elegir. La FOC es $(1-\delta)\,\Pi'(\beta)/(\Pi - \overline{\Pi}) + \delta\,CE'(\beta)/(CE - \overline{CE}) = 0$.
- $\beta$ cumple dos funciones a la vez (incentivos y reparto), se aleja de $\beta^{\text{eff}}$ y se pierde excedente.
- El umbral $\overline{W}$ es la riqueza a la que el $\alpha^*$ del régimen holgado cumple la LL con igualdad.

## 4. Información disponible al pactar

- Públicos y conocidos por ambos: $W_t$, todos los parámetros estructurales, los pagos de desacuerdo y la función de continuación $V_P, V_A$ (equilibrio Markov, estacionario).
- No observable por el principal: el esfuerzo $a_t$. Esa es la fricción de riesgo moral y la razón de que haya $\beta > 0$.
- No conocido por nadie al firmar: $\varepsilon_t$. Por eso el contrato fija $\beta_t$ y el esfuerzo antes de ver el shock.
- No hay información privada previa a la firma (ni selección adversa, ni tipos). La asimetría es solo ex post sobre el esfuerzo.

## 5. Qué se actualiza entre periodos

Una sola cosa: **$W$**, por $W_{t+1} = W_t + \alpha_t + \beta_t y_t - \frac{k}{2}a_t^2$.

- La agente no consume. Toda la paga neta del costo de esfuerzo se acumula como riqueza.
- Todo lo demás es función de $W_t$: $\alpha^*(W)$, $\beta^*(W)$, $\lambda(W)$, $V_P(W)$, $V_A(W)$.
- No se actualizan $\delta$, los pagos de desacuerdo, $\underline{y}$ ni ningún parámetro tecnológico.
- Canal de retroalimentación: $W_t \to \underline{w}(W_t) \to \beta^*_t \to \lambda_t$, y de vuelta por $\beta_t^* \sigma \varepsilon_t \to W_{t+1}$.

## 6. Reglas del bargaining y cómo entra el poder de negociación

Hay dos objetos distintos con nombres parecidos:

- **$\delta$ (primitivo, exógeno)**: el peso de la agente en el producto de Nash. Es constante en el tiempo.
- **$\lambda_t$ (realizado, endógeno)**: la fracción del excedente neto que efectivamente recibe la agente, $\lambda_t = (CE_t - \overline{CE})/\Sigma_t$ con $\Sigma_t$ la suma de los dos excedentes netos.

La relación entre ambos:

- Régimen holgado: $\lambda_t = \delta$.
- Régimen restringido: $\lambda_t \ne \delta$. En la calibración base $\lambda_t > \delta$ y decrece con $W$ hasta llegar a $\delta$ en $\overline{W}$ (Proposición de Apéndice B en el draft).
- Como $W$ se mueve con los shocks de producto, $\lambda_t$ es una variable de estado que evoluciona: shock positivo sube $W$, baja $\lambda$ (cuando $W < \overline{W}$). Esa es la historia de poder de negociación endógeno.

Sobre el protocolo: el Nash bargaining es axiomático. No hay ofertas, contraofertas ni turnos; nadie propone. $\delta$ resume cuánto pesa cada parte sin explicar de dónde sale.

## 7. Ejemplo numérico (calibración base, versión estática)

Parámetros: $\theta = 1$, $k = 2$, $\gamma = 1$, $\sigma = 0.3$, $\delta = 0.4$, $\overline{w} = -0.3$, $\gamma_w = 0.2$, $\underline{y} = -0.5$, desacuerdos en 0. Entonces $\nu = 0.5$, $\kappa = 0.09$, $\beta^{\text{eff}} = 0.8475$, $S^{\text{eff}} = 0.2119$, $\overline{W} = 0.931$.

| $W_t$ | Piso $\underline{w}$ | Régimen | $\beta^*$ | $\alpha^*$ | Esfuerzo | $\lambda$ |
|---|---|---|---|---|---|---|
| 0.2 | -0.340 | restringido | 0.6715 | -0.0043 | 0.336 | 0.435 |
| 0.5 | -0.400 | restringido | 0.7464 | -0.0268 | 0.373 | 0.419 |
| 0.8 | -0.460 | restringido | 0.8175 | -0.0513 | 0.409 | 0.405 |
| 1.2 | -0.540 | holgado | 0.8475 | -0.0625 | 0.424 | 0.400 |

Lectura: una agente pobre recibe menos incentivos ($\beta$ bajo, esfuerzo bajo) y se queda con más que su peso nominal $\delta$. Al enriquecerse converge a $(\beta^{\text{eff}}, \delta)$.

## 8. Huecos e inconsistencias a decidir

Ordenados de más a menos importantes para la definición del juego.

1. **Ley de movimiento de $W$.** La tesis y la teoría usan $W_{t+1} = W_t + w_t - c(a_t)$. El código usa $W_{t+1} = \rho_W W_t + \dots$ con $\rho_W = 0.85$, porque sin depreciación $W$ deriva sin límite en la calibración base (ver `params.py`). Hay que elegir: modelar el sumidero de riqueza (consumo, impuesto, depreciación, retiro) como parte del juego, o mantener $\rho_W = 1$ y cambiar la estructura de la LL para que el proceso sea estacionario.
2. **¿Qué es el desacuerdo en el juego dinámico?** El draft lo define como un pago exógeno si la negociación "se rompe para siempre". En el producto de Nash dinámico se resta $\overline{\Pi}$ (flujo de un periodo) a un objeto que incluye valores de continuación descontados. Eso mezcla unidades; el código lo evita fijando los desacuerdos en 0. Falta decidir: ¿ruptura permanente con valor de por vida $\overline{\Pi}/(1-\rho)$, o ruptura por un periodo con continuación? ¿Depende de $W$?
3. **Dos definiciones de $\lambda$.** El draft (parte estática) define $\lambda$ con pagos de un periodo. La tesis y el código lo definen con pagos que incluyen continuación. Son objetos distintos y las proposiciones del apéndice B están probadas para la versión estática.
4. **Régimen holgado en el problema dinámico.** El código asume que en holgado vale $\beta^{\text{eff}}$ y $\lambda = \delta$ exactamente, y calcula $\overline{W}$ con el $\alpha$ estático. Pero $\alpha$ y $\beta$ mueven $W'$ y con ello $E[V(W')]$, así que no es obvio que la FOC en $\beta$ siga reduciéndose a $S'(\beta) = 0$. Hay que demostrarlo o corregir el umbral.
5. **LL con soporte gaussiano.** La LL se impone en $\underline{y}$, que no es el mínimo real del soporte. Si $y_t < \underline{y}$ el salario cae bajo el piso. No hay regla para ese caso (hoy se paga igual y $W'$ se trunca en los bordes de la grilla). Hay que fijarla o acotar el soporte del shock.
6. **Quién puede abandonar.** No hay regla de salida. Si en algún $W$ el conjunto factible es vacío, el código cae a $\beta^{\text{eff}}$ en vez de terminar la relación. Hay que decidir qué pasa cuando $\Sigma_t \le 0$.
7. **Restricción de $W$.** ¿Puede $W$ ser negativa? ¿Hay endeudamiento? Hoy el dominio es una grilla compacta; no está dicho en el modelo.
8. **Protocolo de bargaining.** Decidir si basta con Nash axiomático con $\delta$ exógeno o si conviene un microfundamento (alternating offers a la Binmore-Rubinstein-Wolinsky) donde $\delta$ salga de paciencia relativa o riesgo de ruptura. Esto afecta cómo se interpreta "poder de negociación".
9. **Notación del timing.** El draft dice "at $t=0$ the public state is observed" dentro de un periodo, y choca con el índice de periodo $t = 0, 1, \dots$. Conviene un índice distinto para las etapas (por ejemplo etapas I a V).
10. **Dinámica ausente del draft.** `bargainigpower_draft.tex` todavía no tiene el entorno dinámico (ley de $W$, Bellman, PMSE); los apéndices usan $W_{t+1} - W_t = \beta_t^* \sigma \varepsilon_t$ sin que esté definida en el texto. Está registrado en `NOTES.md`.
