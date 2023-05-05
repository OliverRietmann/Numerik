---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Woche 11: Differentialgleichung (DGL)

Lernziele:

1. Ich kann ein gegebenes Anfangswertproblem mit der expliziten Eulermethode lösen.
2. Ich kann ein gegebenes Anfangswertproblem mit `scipy.integrate.solve_ivp(...)` lösen.
3. Ich kann eine DGL 2. Ordnung in eine zweidimensionale DGL 1. Ordnung überführen.

## Explizite Eulermethode

Wir lösen das Anfangswertproblem

$$
y'(t)=-\tfrac{1}{2}\cdot y(t),\qquad y(0)=100
$$

mit der expliziten Eulermethode

```{code-cell} ipython3
import numpy as np
import matploylib.pyplot as plt

def explicit_euler(T, y0, n):
    h = T / n
    t = np.linspace(0.0, T, n + 1)
    y = np.empty_like(t)
    y[0] = y0
    for k in range(n):
        y[k + 1] = y[k] + h * (-0.5) * y[k]
    return t, y

T = 10.0
y0 = 100.0

t, y = explicit_euler(T, y, 50)

plt.figure()
plt.plot(t, y, label='explicit Euler')
plt.plot(t, np.exp(-0.5 * t), label='exact')
plt.legend()
plt.show()
```

:::{admonition} Aufgabe
Passen Sie die Funktion `explicit_euler(...)` so an, dass sie ein beliebiges Anfangswertproblem

$$
y'(t)=f(t, y(t)),\qquad y(t_0)=y_0
$$

mit der expliziten Eulermethode lösen kann.
:::
```{code-cell} ipython3
import numpy as np
import matploylib.pyplot as plt

def explicit_euler(f, t_span, y0, n):
    h = (t_span[1] - t_span[0]) / n
    t = np.linspace(t_span[0], t_span[1], n + 1)
    y = np.empty_like(t)
    y[0] = y0
    for k in range(n):
        y[k + 1] = y[k] + h * f(t, y[k])
    return t, y

f = lambda t, y: -0.5 * y
t_span = [0.0, 10.0]
y0 = 100.0

t, y = explicit_euler(T, y, 50)

plt.figure()
plt.plot(t, y, label='explicit Euler')
plt.plot(t, np.exp(-0.5 * t), label='exact')
plt.legend()
plt.show()
```