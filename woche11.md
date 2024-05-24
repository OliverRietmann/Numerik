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

<!--
```{figure} images/euler_polygonzug.png
---
scale: 33%
align: right
---
```
-->

## Explizite Eulermethode

Wir lösen das Anfangswertproblem

$$
\begin{cases}
\quad y'(t) =-t^2\cdot y(t) \\[10pt]
\quad y(0)=200
\end{cases}
$$

mit $n=50$ Schritten der expliziten Eulermethode mit Zeitschrittweite $h=0.05$.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Unten finden wir eine bessere Version dieser Funktion
def explicit_euler(y0, h, n):
    t = np.empty(n + 1)
    t[0] = 0.0
    y = np.empty(n + 1)
    y[0] = y0
    for k in range(n):
        t[k + 1] = t[k] + h
        y[k + 1] = y[k] + h * (-t[k]**2 * y[k])
    return t, y

y0 = 200.0
h = 0.05
n = 50

t, y = explicit_euler(y0, h, n)

plt.figure()
plt.plot(t, y0 * np.exp(-t**3 / 3.0), 'r-', label='exact')
plt.plot(t, y, 'c--', label='explicit Euler')
plt.legend()
plt.show()
```

:::{admonition} Aufgabe
Der Code oben funktioniert nur für eine spezielle Differentialgleichung und nur für die Startzeit $t_0=0$.
Passen Sie die Funktion `explicit_euler(...)` so an, dass sie ein beliebiges Anfangswertproblem

$$
y^\prime(t)=f(t, y(t)),\qquad y(t_0)=y_0
$$

mit der expliziten Eulermethode lösen kann.
:::

Lösung:
```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

def explicit_euler(f, t0, y0, h, n):
    t = np.empty(n + 1)
    t[0] = t0                                # geändert
    y = np.empty(n + 1)
    y[0] = y0
    for k in range(n):
        t[k + 1] = t[k] + h
        y[k + 1] = y[k] + h * f(t[k], y[k])  # geändert
    return t, y

f = lambda t, y: -t**2 * y
t0 = 0.0
y0 = 200.0
h = 0.05
n = 50

t, y = explicit_euler(f, t0, y0, h, n)

plt.figure()
plt.plot(t, y0 * np.exp(-t**3 / 3.0), 'r-', label='exact')
plt.plot(t, y, 'c--', label='explicit Euler')
plt.legend()
plt.show()
```

## Zweidimensionale DGL

Wir lösen das Anfangswertproblem

$$
\begin{pmatrix}
y_1^\prime(t) \\
y_2^\prime(t)
\end{pmatrix}=
\begin{pmatrix}
-0.5 & 0 \\
0.5 & -0.2
\end{pmatrix}\cdot
\begin{pmatrix}
y_1(t) \\
y_2(t)
\end{pmatrix},\qquad 
\begin{pmatrix}
y_1(0) \\
y_2(0)
\end{pmatrix}=
\begin{pmatrix}
100\\
100
\end{pmatrix}
$$

mit der expliziten Eulermethode.
Die Funktion `explicit_euler(...)` bleibt fast unverändert.
```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

def explicit_euler(f, t0, y0, h, n):
    t = np.empty(n + 1)
    t[0] = t0
    y = np.empty((n + 1, len(y0)))  # geändert
    y[0] = y0
    for k in range(n):
        t[k + 1] = t[k] + h
        y[k + 1] = y[k] + h * f(t[k], y[k])
    return t, y

M = np.array([[-0.5, 0.0], [0.5, -0.2]])
f = lambda t, y: np.dot(M, y)
t0 = 0.0
y0 = np.array([100.0, 100.0])
h = 0.1
n = 200

t, y = explicit_euler(f, t0, y0, h, n)

plt.figure()
plt.title('Radioaktive Zerfallskette (Produkt 1 zerfällt weiter in Produkt 2)')
plt.plot(t, y[:, 0], label='Zerfallsprodukt 1')
plt.plot(t, y[:, 1], label='Zerfallsprodukt 2')
plt.legend()
plt.show()
```

## Automatische Wahl der Zeitschritte

Wir lösen das Anfangswertproblem

$$
\begin{pmatrix}
z_1^\prime(t) \\
z_2^\prime(t)
\end{pmatrix}=
\begin{pmatrix}
0 & 1 \\
-1 & 0
\end{pmatrix}\cdot
\begin{pmatrix}
z_1(t) \\
z_2(t)
\end{pmatrix},\qquad 
\begin{pmatrix}
z_1(0) \\
z_2(0)
\end{pmatrix}=
\begin{pmatrix}
1\\
0
\end{pmatrix}
$$

mit der Funktion `numpy.integrate.solve_ivp(...)` bis zur Endzeit $T=2\pi$
```{code-cell} ipython3
import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'scipy'])

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

M = np.array([[0.0, 1.0], [-1.0, 0.0]])
f = lambda t, y: np.dot(M, y)
t_span = [0.0, 2.0 * np.pi]  # Start- und Endzeit
n = 100                      # Anzahl Zeitschritte
z0 = np.array([1.0, 0.0])

t_eval = np.linspace(t_span[0], t_span[1], n + 1, endpoint=True)
sol = sp.integrate.solve_ivp(f, t_span, z0, t_eval=t_eval)
t = sol.t
y = sol.y

plt.figure()
plt.title('Pendulum: Space-Time')
plt.plot(t, y[0], label='position')
plt.plot(t, y[1], label='velocity')
plt.legend()
plt.show()

plt.figure()
plt.title('Pendulum: Phasespace')
plt.plot(y[0], y[1])
plt.show()
```
