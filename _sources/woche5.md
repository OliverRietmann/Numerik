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

# Woche 5: Newton-Verfahren (Vertiefung)

Lernziele:

1. Ich kann die Jacobi-Matrix einer Funktion von $\mathbb R^n$ nach $\mathbb R^n$ berechnen, wobei $1\leq n\leq 3$.
2. Ich kann das Newton-Verfahren in 1D und 2D implementieren.
3. Ich kann die Funktion `matplotlib.pyplot.plot_surface` verwenden.

## Minimierung einer differenzierbaren Funktion

Wir betrachten die Funktion $f(x)=x^2\sin(x)$ im Intervall $[2,7]$.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: x**2 * np.sin(x)
df = lambda x: 2.0 * x * np.sin(x) + x**2 * np.cos(x)
x = np.linspace(2.0, 7.0, 100)

fig, axes = plt.subplots(2)
axes[0].plot(x, f(x), label=r'$f(x)$')
axes[1].plot(x, df(x), label=r'$f^\prime(x)$')
for ax in axes:
    ax.legend()
    ax.grid(True)
plt.show()
```

:::{admonition} Aufgabe
Finden Sie das globale Minimum obiger
Funktion mit dem Newton-Verfahren.
Ergänzen Sie dazu folgenden Code.
:::
```{code-cell} ipython3
import numpy as np

f = lambda x: x**2 * np.sin(x)
df = lambda x: 2.0 * x * np.sin(x) + x**2 * np.cos(x)
ddf = lambda x: (2.0 - x**2) * np.sin(x) + 4.0 * x * np.cos(x)

def newton(g, dg, x, tol):
    # Ihr Code kommt hier hin.
    # ...
    return x

tol = 1.0e-5
for x0 in [3.0, 4.0, 5.0]:
    # Ihr Code kommt hier hin.
    # ...
    pass
```
<!--
import numpy as np

f = lambda x: x**2 * np.sin(x)
df = lambda x: 2.0 * x * np.sin(x) + x**2 * np.cos(x)
ddf = lambda x: (2.0 - x**2) * np.sin(x) + 4.0 * x * np.cos(x)

def newton(g, dg, x, tol):
    while np.abs(g(x)) > tol:
        x = x - g(x) / dg(x)
    return x

tol = 1.0e-5
for x0 in [3.0, 4.0, 5.0]:
    print(newton(df, ddf, x0, tol))
-->
## Plotten einer Funktion von $\mathbb R^2$ nach $\mathbb R$.

Wir plotten die Funktionen

$$
f_1(x, y)=2x+4y
\quad\text{und}\quad
f_2(x, y)=4x+8y^3
$$

mit `matplotlib.pyplot.plot_surface`.
Dazu müssen diese auf einem `numpy.meshgrid` ausgewertet werden.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5.0, 5.0, 50)
y = np.linspace(-5.0, 5.0, 50)
X, Y = np.meshgrid(x, y)

f1 = lambda x, y: 2.0 * x + 4.0 * y
f2 = lambda x, y: 4.0 * x + 8.0 * y**3

fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
ax1.title.set_text(r'$f_1(x, y)$')
ax1.plot_surface(X, Y, f1(X, Y))
ax2.title.set_text(r'$f_2(x, y)$')
ax2.plot_surface(X, Y, f2(X, Y))
plt.show()
```

## Mehrdimensionales Newton-Verfahren

Wir betrachten wieder die Funktion

$$
f(x, y)=
\begin{pmatrix}
f_1(x, y) \\
f_2(x, y)
\end{pmatrix}=
\begin{pmatrix}
2x+4y \\
4x+8y^3
\end{pmatrix}.
$$

Wir wollen folgendes System nicht-linearer Gleichungen lösen

$$
\begin{pmatrix}
0 \\
0
\end{pmatrix}
=f(x, y).
$$

Das Newton-Verfahren lautet nun

$$
\begin{pmatrix}
x_{k+1} \\
y_{k+1}
\end{pmatrix}=
\begin{pmatrix}
x_k \\
y_k
\end{pmatrix}
-\left(J(x_k, y_k)\right)^{-1}\cdot f(x_k, y_k),
\qquad
J(x, y)=
\begin{pmatrix}
2 & 4 \\
4 & 24y^2
\end{pmatrix}.
$$

Hier bezeichnet $J(x, y)$ die Jacobi-Matrix von $f$ an der Stelle $(x, y)$.
Die Matrix-Vektor Multiplikation entspricht dem Lösen eines linearen Gleichungssystems (LGS):

$$
\vec{d}=\left(J(x, y)\right)^{-1}\cdot f(x, y)
\quad\Longleftrightarrow\quad
J(x, y)\cdot\vec{d}=f(x, y).
$$

:::{admonition} Aufgabe
Ergänzen Sie folgenden Code zu einem Newton-Verfahren das eine Nullstelle von $f$ berechnet.
Lösen Sie das LGS mit `numpy.linalg.solve(J(x, y), f(x, y))`.
:::
```{code-cell} ipython3
import numpy as np

f1 = lambda x, y: 2.0 * x + 4.0 * y
f2 = lambda x, y: 4.0 * x + 8.0 * y**3
f = lambda x, y: np.array([f1(x, y), f2(x, y)])
J = lambda x, y: np.array([[2.0, 4.0], [4.0, 24.0 * y**2]])

def newton(f, J, x, y, tol, N):
    n = 0
    # Ihr Code kommt hier hin
    # ...
    # ...
    return x, y, n

x0, y0 = (3.0, 2.0)
print(newton(f, J, x0, y0, 1.0e-3, 20))
```
<!--
import numpy as np

f1 = lambda x, y: 2.0 * x + 4.0 * y
f2 = lambda x, y: 4.0 * x + 8.0 * y**3
f = lambda x, y: np.array([f1(x, y), f2(x, y)])
J = lambda x, y: np.array([[2.0, 4.0], [4.0, 24.0 * y**2]])

def newton(f, J, x, y, tol, N):
    n = 0
    while np.linalg.norm(f(x, y)) > tol and n < N:
        d = np.linalg.solve(J(x, y), f(x, y))
        x, y = np.array([x, y]) - d
        n += 1
    return x, y, n

x0, y0 = (3.0, 2.0)
print(newton(f, J, x0, y0, 1.0e-3, 20))
```
-->

Die Nullstellen von $f$ sind

$$
\begin{pmatrix}
0 \\
0
\end{pmatrix},\qquad
\begin{pmatrix}
2 \\
-1
\end{pmatrix},\qquad
\begin{pmatrix}
-2 \\
1
\end{pmatrix}.
$$
<!--
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-3.0, 3.0, 50)
y = np.linspace(-3.0, 3.0, 50)
X, Y = np.meshgrid(x, y)

f1 = lambda x, y: 2.0 * x + 4.0 * y
f2 = lambda x, y: 4.0 * x + 8.0 * y**3
residuum = lambda x, y: np.sqrt(f1(x, y)**2 + f2(x, y)**2)

plt.figure()
plt.title("Residuum")
plt.contourf(X, Y, residuum(X, Y))
plt.show()
```
-->