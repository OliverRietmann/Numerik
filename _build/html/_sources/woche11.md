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
\begin{cases}
y_0^\prime(t)&=-0.5\cdot y_0(t) \\
y_1^\prime(t)&=\phantom{-}0.5\cdot y_0(t)-0.2\cdot y_1(t) \\[5pt]
y_0(0)&=100 \\
y_1(0)&=100
\end{cases}
$$

mit der expliziten Eulermethode.
Man kann (muss aber nicht) diese DGL mit einer Matrix schreiben:

$$
\begin{pmatrix}
y_0^\prime(t) \\
y_1^\prime(t)
\end{pmatrix}=
\begin{pmatrix}
-0.5 & 0 \\
0.5 & -0.2
\end{pmatrix}\cdot
\begin{pmatrix}
y_0(t) \\
y_1(t)
\end{pmatrix},\qquad 
\begin{pmatrix}
y_0(0) \\
y_1(0)
\end{pmatrix}=
\begin{pmatrix}
100\\
100
\end{pmatrix}
$$

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
# alternativ: f = lambda t, y: np.array([-0.5 * y[0], 0.5 * y[0] - 0.2 * y[1]])
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
z_0^\prime(t) \\
z_1^\prime(t)
\end{pmatrix}=
\begin{pmatrix}
0 & 1 \\
-1 & 0
\end{pmatrix}\cdot
\begin{pmatrix}
z_0(t) \\
z_1(t)
\end{pmatrix},\qquad 
\begin{pmatrix}
z_0(0) \\
z_1(0)
\end{pmatrix}=
\begin{pmatrix}
1\\
0
\end{pmatrix}
$$

mit der Funktion `scipy.integrate.solve_ivp(f, t_span, y0)` bis zur Endzeit $T=2\pi$.
Diese Funktion löst das Anfangswertproblem (**i**nitial **v**alue **p**roblem) auf dem Zeitintervall `t_span`, wobei `f(t, y)` dessen rechte Seite ist.
Dabei muss `y` (auch im 1D Fall) ein Vektor sein.
Dasselbe gilt für den Anfangswert `y0`.

```{code-cell} ipython3
import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'scipy'])

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

f = lambda t, z: np.array([z[1], -z[0]])
t_span = [0.0, 2.0 * np.pi]  # Start- und Endzeit
n = 100                      # Anzahl Zeitschritte
z0 = np.array([1.0, 0.0])

sol = sp.integrate.solve_ivp(f, t_span, z0)
t = sol.t  # Das sind die von solve_ivp(...) selbst generierten Zeiten
y = sol.y  # Das sind die z(t)-Werte zu diesen Zeiten

t_exact = np.linspace(*t_span)

plt.figure()
plt.title('Pendulum: approx vs. exact')
plt.plot(t, y[0], 'c-', label=r'$z_0(t)$ approx')
plt.plot(t_exact, np.cos(t_exact), 'c--', label=r'$z_0(t)$ exact')
plt.plot(t, y[1], 'b-', label=r'$z_1(t)$ approx')
plt.plot(t_exact, -np.sin(t_exact), 'b--', label=r'$z_1(t)$ exact')
plt.legend()
plt.show()
```
