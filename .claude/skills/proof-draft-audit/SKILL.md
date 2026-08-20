---
name: proof-draft-audit
description: Redacta una demostración matemática formal y la somete a un ciclo RECURSIVO de auditoría estilo referee de journal top (Econometrica/AER/JET) — redactor propone, auditor da retroalimentación, redactor revisa e incorpora mejoras, auditor vuelve a revisar — hasta convergencia (aceptación sin objeciones sustantivas) o hasta un tope de rondas. No son agentes que dialogan en tiempo real; cada ronda es una pasada de razonamiento desacoplada de la anterior. Úsalo cuando el usuario pida "demuestra que...", "formaliza este argumento", "redacta la prueba de...", "necesito una demostración auditada/verificada a nivel referee", o pida explícitamente el flujo "redactor y auditor" para una proposición, lema, o resultado matemático. También aplica cuando el usuario da un argumento informal o una intuición y pide que se convierta en demostración formal con verificación rigurosa.
---

# Redactor-Auditor de Demostraciones (ciclo recursivo)

Flujo iterativo para producir demostraciones matemáticas al nivel de un journal top en economía teórica/econometría: el Redactor propone, el Auditor —con estándar de referee de Econometrica/AER/JET— da retroalimentación, el Redactor revisa e itera, y así sucesivamente hasta que el Auditor no tenga objeciones sustantivas nuevas (convergencia) o se alcance el tope de rondas.

Ambos roles los ejecuta Claude en la misma conversación, pero **como pasadas de razonamiento separadas y desacopladas por ronda** — en cada ronda, el Auditor relee el enunciado y la versión actual de la demostración como lo haría un revisor externo que recibe el manuscrito, no como quien "recuerda" haberla escrito. El Auditor sí puede (y debe) revisar si sus objeciones de rondas anteriores fueron atendidas, pero audita el texto tal como quedó, no la intención detrás de él.

## Cuándo usar este skill

- El usuario pide demostrar un resultado (a partir de un enunciado que él da), **o**
- El usuario da un argumento informal / una intuición y pide que se formalice, **o**
- El usuario pide explícitamente el flujo "redactor + auditor" sobre una demostración ya escrita (en ese caso la Ronda 1 empieza directo en la auditoría de lo que él proporcionó, sin fase de redacción inicial).

No uses este skill para verificaciones triviales de una línea (esas se resuelven directo, sin el ceremonial de rondas).

## Idioma y estilo

Todo el output es en español, con conjugación tú/usted — **nunca voseo** (nada de "vos", "recordá", "opinás", "imaginás"). Notación matemática en símbolos estándar. Si el usuario pide o cita en inglés, se puede citar en inglés, pero el cuerpo del texto se mantiene en español salvo instrucción contraria.

## Fase 0 — Insumo

Antes de empezar, confirma (o infiere razonablemente del contexto) y anúncialo en una línea antes de arrancar:

1. **El enunciado exacto a demostrar.** Si el usuario dio una intuición informal, el primer paso del Redactor en la Ronda 1 es **formalizar el enunciado** (objetos, hipótesis, cuantificadores) antes de demostrarlo — esa formalización es en sí misma auditable y debe quedar explícita.
2. **El marco / convenciones a usar** (p. ej. notación de Kamenica-Gentzkow, convenciones de KTG/FD, u otro paper del proyecto). Si el resultado vive dentro de uno de los papers del proyecto, usa Grep/Read (o el agente Explore) sobre los archivos del repositorio para confirmar notación y definiciones antes de redactar.
3. **Tope de rondas.** Por defecto, máximo **5 rondas**. Si al llegar a la ronda 5 no hay convergencia, detente y entrega un resumen explícito de qué objeciones quedan abiertas — nunca sigas iterando indefinidamente ni finjas convergencia.

## Fase 0.5 — Anclaje de contexto y registro económico

**Esto es obligatorio, no opcional, y va antes de escribir una sola línea de la demostración.** El riesgo específico a evitar: que el modelo redacte una prueba matemáticamente correcta pero "descontextualizada" — que hable de funciones, dominios y desigualdades genéricas en vez de tipos, creencias, Emisor/Receptor y condiciones de no-desviación. Esto es un error de fondo, no de estilo: en un juego de señalización, la demostración *es* sobre el comportamiento estratégico, y la manipulación algebraica es el vehículo, no el contenido.

Antes de redactar:

1. Usa Grep/Read (o el agente Explore) para localizar el resultado específico (o el paper del que proviene) dentro del repositorio y extrae: qué notación usa para tipos/señales/acciones, cómo llama a las creencias (posterior, µ\*, condición de Bayes-plausibilidad), y cómo estructura los argumentos de no-desviación (IC, "deterrence", indiferencia).
2. Si el resultado no vive en un paper específico del proyecto, usa como referencia por defecto el registro de Kamenica-Gentzkow (persuasión bayesiana: experimento, valor cóncavo/envolvente, creencia posterior) o el vocabulario estándar de juegos de señalización (tipo, mensaje/señal, Emisor, Receptor, equilibrio de agrupación/separador/semi-separador, PBE, D1, Criterio Intuitivo) — nunca el registro de "análisis puro" (función, dominio, óptimo) sin ese anclaje.
3. Consulta `references/registro_economico.md` (en este skill) para ejemplos concretos del vocabulario esperado tomado de tus propios papers.

Anuncia en una línea qué convención estás usando antes de pasar a la Ronda 1.

## Nota sobre independencia real del Auditor

Dentro de una misma conversación, el Auditor nunca es completamente independiente del Redactor —
aunque relea el texto "desde cero", sigue en la misma sesión que acaba de producirlo, con el mismo
contexto flotando alrededor. Para trabajo donde la independencia del auditor importa de verdad
(una demostración que va a entrar al paper, no un ejercicio exploratorio), la forma correcta de
usar este skill es: pedir la demostración en una conversación **nueva**, sin el historial de haberla
redactado antes. Si vienes de una conversación donde ya se discutió el resultado o se intentó una
versión previa, considera empezar una conversación aparte para la corrida "oficial" del skill.

## El ciclo

### Ronda 1 — Redactor

Objetivo: la demostración más rigurosa y completa posible en el primer intento, sin pensar todavía en cómo será criticada.

- Declara hipótesis, definiciones y notación antes de usarlas.
- Cada paso justificado (regla de inferencia, sustitución algebraica, lema citado) — nunca "se sigue que" sin decir por qué.
- Casos cubiertos explícitamente, sin dejar "obvios" sin tratar.
- Verifica algebraicamente con sympy (aritmética exacta, `Fraction`) cualquier paso no trivial; deja constancia en una sección "Verificación computacional" (código + resultado).
- Cierra con QED y, si aplica, el alcance/condiciones bajo las que vale el resultado.

Encabezado: `### Demostración — Ronda 1 (Redactor)`.

### Auditoría — cada ronda (Auditor)

**Estándar: referee de journal top de economía teórica/econometría (Econometrica, AER, JET, RESTUD).** Esto es más exigente que una revisión de corrección matemática: un referee de ese nivel evalúa cinco dimensiones, y las cinco son motivo de otra ronda si fallan:

1. **Corrección matemática** — el paso es válido, la desigualdad se sostiene, los casos están bien cubiertos. (Categoría: *error matemático* — invalida el resultado.)
2. **Rigor y notación** — cuantificadores sin ambigüedad, notación consistente con el resto del documento/paper, distinción cuidadosa entre "si y solo si", "se reduce a", e "implica" (no tratarlas como sinónimos). (Categoría: *rigor/notación*.)
3. **Cohesión argumental y orden de presentación** — ¿la prueba fluye en el orden que más ayuda al lector, o hay pasos fuera de lugar? ¿Cada lema/paso se usa donde se necesita, o se introduce material antes de que quede claro para qué sirve? ¿La estructura general (por contradicción, inducción, casos, construcción directa) es la más transparente para este resultado, o hay una reorganización que la haría más legible? (Categoría: *cohesión/estructura*.)
4. **Claridad expositiva** — frases puente entre pasos, resultados citados explícitamente, ausencia de saltos que un lector experto pero no omnisciente tendría que reconstruir por su cuenta. (Categoría: *claridad expositiva*.)
5. **Registro económico** — ¿la demostración está enunciada en el vocabulario de teoría de juegos que corresponde (tipos, creencias, Bayes-plausibilidad, Emisor/Receptor, condiciones de no-desviación, pooling/separating/semi-separating, refinamientos con nombre) o degeneró en lenguaje de análisis puro genérico (funciones, dominios, "óptimo" sin decir de quién ni por qué es mejor respuesta)? Un paso puede ser matemáticamente impecable y aun así fallar aquí si pierde la interpretación estratégica — eso cuenta como objeción sustantiva, igual que un error matemático. (Categoría: *registro económico*.) Verifica en particular que cada paso algebraico venga acompañado de su lectura económica (qué tipo se desvía, qué creencia se sostiene, qué indiferencia se está imponiendo), no solo la manipulación formal.

Para cada hallazgo: cita el paso exacto, explica el problema, clasifícalo en una de las cinco categorías, y si es un error matemático, **reverifica independientemente con sympy** desde el enunciado (no repitas el cálculo del Redactor).

Cierra cada auditoría con un veredicto explícito de tres posibles valores — este veredicto es el que controla si el ciclo continúa:

- **CONVERGE** — no hay objeciones sustantivas (matemáticas, de rigor, estructurales, o de registro económico) pendientes; lo que quede, si algo, son preferencias estilísticas menores y opcionales que no ameritan otra ronda.
- **OTRA RONDA** — hay al menos una objeción sustantiva (cualquiera de las 5 categorías) que el Redactor debe atender.
- **TOPE ALCANZADO** — solo se usa en la ronda del límite (por defecto ronda 5) si aún hay objeciones sustantivas; no se finge convergencia.

Encabezado: `### Auditoría — Ronda N (Auditor)`.

### Redactor — rondas 2 en adelante

- Atiende cada objeción de la auditoría anterior explícitamente (no las ignores ni las diluyas).
- Además de corregir lo señalado, **aprovecha la relectura para mejorar lo que el propio Redactor note** en esta ronda, aunque el Auditor no lo haya marcado — igual que un autor que, al revisar por objeciones de un referee, también pule otras partes que ve mejorables.
- Vuelve a verificar con sympy cualquier paso que haya cambiado.
- Encabezado: `### Demostración — Ronda N (Redactor)`.

### Condición de paro

El ciclo se detiene en la primera ronda en que ocurra cualquiera de:
- El Auditor emite **CONVERGE**.
- Se alcanza el tope de rondas (por defecto 5) sin CONVERGE — en ese caso el estado final es **TOPE ALCANZADO**, y se listan las objeciones que quedaron sin resolver.

No hay una fase separada de "el redactor corrige al auditor" fuera de este ciclo — toda corrección ocurre dentro de las rondas descritas arriba. Si tras la entrega el usuario quiere una ronda adicional manual, es una solicitud nueva y explícita, no algo automático.

## Entrega

Guarda el resultado como un archivo `.md` **por demostración**, con esta estructura interna (todas las rondas quedan documentadas, no solo la final — el historial de convergencia es parte del valor del ejercicio):

```markdown
# [Nombre corto de la demostración/proposición]

## Enunciado
[enunciado formal, con hipótesis y notación explícitas]

## Ronda 1
### Demostración — Ronda 1 (Redactor)
[...]
### Auditoría — Ronda 1 (Auditor)
[hallazgos por categoría + reverificación sympy]
**Veredicto:** OTRA RONDA / CONVERGE

## Ronda 2
### Demostración — Ronda 2 (Redactor)
[cambios respecto a la ronda anterior, explícitos]
### Auditoría — Ronda 2 (Auditor)
**Veredicto:** ...

[... rondas adicionales si aplica ...]

## Versión final
[la demostración de la última ronda, limpia, sin marcas de revisión]

## Resumen de convergencia
- Rondas totales: N
- Estado final: CONVERGE / TOPE ALCANZADO
- Principales cambios entre Ronda 1 y la versión final
- (si TOPE ALCANZADO) objeciones que quedaron abiertas
```

Nombra el archivo de forma descriptiva, p. ej. `demostracion_proposicion4_ktg.md`. Guárdalo con la herramienta Write en el directorio de trabajo actual (o donde el usuario indique); no hay equivalente de `present_files` en Claude Code, así que basta con confirmar al usuario la ruta final del archivo. Si el resultado vive dentro de un proyecto con notación establecida (KTG, FD, attacker-defender), usa esa notación consistentemente en todas las rondas.

## Notas sobre el ciclo

- Cada ronda del Auditor debe ser genuinamente independiente en su verificación algebraica — no basta con confirmar visualmente que el Redactor "atendió" el punto; hay que reconstruir la verificación.
- El Auditor no debe inflar objeciones para forzar rondas adicionales artificialmente, ni desinflarlas para converger rápido: el criterio de las 5 categorías es el que decide, no una meta de número de rondas.
- Si dos rondas consecutivas del Auditor señalan exactamente los mismos puntos sin que el Redactor los haya resuelto, es señal de un problema real (posible imposibilidad de reparar el paso con el enfoque actual) — en ese caso, la siguiente ronda del Redactor debe considerar explícitamente un cambio de estrategia de prueba, no solo un parche más de lo mismo.
