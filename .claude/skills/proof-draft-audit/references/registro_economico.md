# Registro económico esperado — ejemplos de referencia

Este archivo cataloga el vocabulario y las convenciones que el Redactor y el Auditor deben usar por defecto, tomado directamente de los papers del proyecto (KTG, FD, Kamenica-Gentzkow). El objetivo es que una demostración de teoría de juegos se *lea* como teoría de juegos, no como análisis real disfrazado de teoría de juegos.

## Vocabulario mínimo esperado (con su versión "genérica" a evitar)

| Concepto | Registro económico (usar) | Registro genérico (evitar) |
|---|---|---|
| El agente informado | Emisor / Sender, tipo θ | "el agente", "la función" |
| El agente que responde | Receptor / Receiver | "el optimizador", "el jugador 2" sin más |
| Distribución sobre tipos tras la señal | creencia posterior µ, condición de Bayes-plausibilidad | "el valor de la función en ese punto" |
| Punto donde el Receptor cambia de mejor respuesta | µ\* (umbral), kink | "punto crítico", "discontinuidad" sin más |
| Verificar que nadie quiere desviarse | condición de no-desviación / incentive compatibility (IC), indiferencia | "condición de optimalidad" a secas |
| Creencia tras una señal fuera de la trayectoria de equilibrio | creencia off-path, P(Y\|B) | "condición de frontera" |
| Clasificación de equilibrios | pooling / separating / semi-separating, equilibrio completamente mixto (CME) | "solución tipo 1", "caso A" sin nombrarlo |
| Refinamiento de equilibrio | D1, Criterio Intuitivo | "el equilibrio razonable" |
| Comparación de valor de experimentos (Kamenica-Gentzkow) | cuerda (chord), envolvente cóncava, valor del experimento | "la envolvente superior de la función" sin decir de qué experimento |

## Ejemplos reales (tomados de los papers, para calibrar el nivel de detalle económico esperado)

**De "Uniqueness of Completely Mixed Equilibria"** — nótese cómo cada paso algebraico se ancla en una interpretación de juego, no solo en un cálculo:

> "Under (S)'s BY > AY, identity AY = BY fails; under (C1)'s AY < B′Y, identity AY = B′Y fails... The remaining identity A′Y = BY is a single algebraic equation on payoffs; it defines a codimension-one hyperplane in payoff space, and the same identity would need to hold for type Z's indifference."

Aquí la condición algebraica (una identidad entre pagos) está explícitamente ligada a *la indiferencia de un tipo entre señales* — no aparece como una ecuación abstracta.

**De KTG (manuscript_main_round3.5_36)** — la lógica de necesidad de una proposición está organizada por qué condición del juego falla, no por qué desigualdad falla:

> "If (C1) fails (AY ≥ B′Y): pooling on A at off-path P(Y | B) < µ∗B has type Y dev payoff B′Y ≤ AY (stays)... Coexisting equilibrium."

El argumento nombra explícitamente el tipo, la señal, la creencia fuera de la trayectoria, y la conclusión estratégica (por qué ese tipo no se desvía) — no solo el signo de una desigualdad.

**De KTG (equilibrios alternativos)**:

> "A pooling on A equilibrium exists for all P(Y) ∈ [0, 1] with the out-of-equilibrium belief P(Y | B) ≥ 2/3, with the Receiver playing C after A and D after B."

Cada objeto matemático (el intervalo, la desigualdad) está atado a un objeto del juego (tipo de equilibrio, creencia fuera de la trayectoria, acción del Receptor tras cada señal).

## Regla práctica para el Auditor

Si al leer un paso de la demostración no queda claro **qué tipo está actuando, qué cree el Receptor en ese punto, o por qué esa desigualdad corresponde a una condición de no-desviación o de Bayes-plausibilidad**, el paso necesita reescritura — aunque el álgebra esté perfecta. Esa es la objeción de la categoría "registro económico", y es tan válida para forzar OTRA RONDA como un error matemático.
