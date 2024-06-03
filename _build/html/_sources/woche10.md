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

# Woche 10: Quadratur

Lernziele:

1. Ich kann zu gegeben Knoten und Gewichten die zugerhörige Quadraturregel implementieren.
2. Ich kann mit `numpy.trapz(...)` eine zusammengesetzte Trapezregel anwenden.
3. Ich kann mit `scipy.integrate.simpson(...)` eine zusammengesetzte Simpsonregel anwenden.

## Allgemeine Quadraturregel

Zu gegebenen Knoten $x_0,\ldots,x_n$ und Gewichten $w_0,\ldots,w_n$ kann man die zugehörige Quadraturregel als Skalarprodukt von Vektoren schreiben

$$
f(x_0)\cdot w_0+f(x_1)\cdot w_1+\ldots+f(x_n)\cdot w_n=
\begin{pmatrix}
    f(x_0) \\
    f(x_1) \\
    \vdots \\
    f(x_n)
\end{pmatrix}
\cdot
\begin{pmatrix}
    w_0 \\
    w_1 \\
    \vdots \\
    w_n \\
\end{pmatrix}
$$

Als Beispiel approximieren wir das Integral

$$
\int_0^3\sin(x)dx
$$

mit den Gewichten der $3/8$ Regel aus der Vorlesung.

```{code-cell} ipython3
import numpy as np

a = 0.0
b = 3.0

# Gewichte der 3/8 Regel (siehe Handout der Vorlesung)
w = (b - a) / 8.0 * np.array([1.0, 3.0, 3.0, 1.0])

n = len(w) - 1
x = np.linspace(a, b, n + 1)

# Die Quadratur ist gerade folgendes Skalarprodukt:
print(np.dot(np.sin(x), w))
```

## Zusammengesetzte Trapezregel

Wir approximieren das Integral

$$
\int_0^3\sin(x)dx.
$$

mit der aus $n=3$ Teilintervallen zusammengesetzten Trapezregel.
Allgemein braucht die aus $n$ Teilintervallen zusammengesetzte Trapezregel genau $n+1$ Quadraturpunkte.

![trapez](images/trapez.png)

In Python geht das mit der Funktion `numpy.trapz(...)`.

```{code-cell} ipython3
import numpy as np

a = 0.0
b = 3.0
n = 3

x = np.linspace(a, b, n + 1, endpoint=True)
y = np.sin(x)

print(np.trapz(y, x))
```

## Zusammengesetzte Simpsonregel

Als Beispiel approximieren wir folgendes Integral mit der aus $n=2$ Teilintervallen zusammengesetzten Simpsonregel:

$$
\int_0^3\sin(x)dx
$$

Allgemein braucht die aus $n$ Teilintervallen zusammengesetzte Simpsonregel genau $2n+1$ Quadraturknoten.
In Python geht das mit der Funktion `scipy.integrate.simpson(...)`.

```{code-cell} ipython3
import subprocess
import sys
subprocess.call([sys.executable, '-m', 'pip', 'install', 'scipy'])

import numpy as np
import scipy as sp

a = 0.0
b = 3.0
n = 2

x = np.linspace(a, b, 2 * n + 1, endpoint=True)
y = np.sin(x)

print(sp.integrate.simpson(y, x=x))
```
