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
3. Ich kann `scipy.integrate.simpson(...)` eine zusammengesetzte Simpsonregel anwenden.
4. Ich kann `scipy.integrate.quad(...)` eine gegebene Funktion numerisch integrieren.

## Allgemeine Quadraturregel

Zu gegebenen Knoten $x_0,\ldots,x_n$ und Gewichten $w_0,\ldots,w_n$ kann man die zugehörige Quadraturregel als Skalarprodukt von Vektoren schreiben

$$
f(x_0)w_0+f(x_1)w_1+\ldots+f(x_n)w_n=
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

So kann man die Quadratur in Python implementieren.

```{code-cell} ipython3
import numpy as np

a = 0
b = np.pi

# Gewichte der 3/8 Regel (siehe Handout der Vorlesung)
w = (b - a) / 8.0 * np.array([1.0, 3.0, 3.0, 1.0])

n = len(w) - 1
x = np.linspace(a, b, n + 1)

# Die Quadratur ist gerade folgendes Skalarprodukt:
print(np.dot(np.sin(x), w))
```

## Trapezregel

Mit der aus $n$ Knoten zusammengesetzten Trapezregel

$$
\frac{h}{2}\cdot\big(f(a)+2f(a+h)+\ldots+2f(a+h(n-1))+f(b)\big),\quad
h=\frac{b-a}{n}
$$

auf dem Intervall $[a,b]$ approximieren wir das Integral

$$
\int_0^\pi\sin(x)dx.
$$

In Python geht das mit der Funktion `numpy.trapz(...)`.

```{code-cell} ipython3
import numpy as np

a = 0
b = np.pi
n = 100

x = np.linspace(a, b, n + 1)
y = np.sin(x)

print(np.trapz(y, x))
```