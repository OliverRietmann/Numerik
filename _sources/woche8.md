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

# Woche 8: Ausgleichsrechnung (Regression)

Lernziele:

1. Ich kann mit `numpy.linalg.solve(...)` ein LGS numerisch lösen.
2. Ich kann erklären, wann und warum der Gauss-Algorithmus schlechte Lösungen liefert.

## Gerade

Wir wollen eine Gerade $f(x)=mx+b$ an gegebene Punkte $(x_i,y_i),i=1,\ldots,n$ fitten.
Die zugehörige Normalengleichung lautet
$$
A^TA\cdot
\begin{pmatrix}
    b \\
    m
\end{pmatrix}
=A^Ty
$$
wobei
$$
A:=
\begin{pmatrix}
    1 & x_1 \\
    1 & x_2 \\
    \vdots & \vdots \\
    1 & x_m
\end{pmatrix},\qquad
y:=
\begin{pmatrix}
    y_1 \\
    y_2 \\
    \vdots \\
    y_m
\end{pmatrix}.
$$

Wir lösen die Normalengleichung mit 'numpy.linalg.solve(...)'.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

n = 4
x = np.linspace(-2.5, 2.5, n)
y = 2.0 * x + 3.0 + 0.5 * np.random.rand(n)

A = np.column_stack((np.ones_like(x), x, x**2))
AT = np.transpose(A)

print(A)

b, m = np.linalg.solve(AT @ A, np.dot(A, y))

plt.figure()
plt.plot(x, y, 'bo')
plt.plot(x, m * x + b, 'r-')
plt.show()
plt.clear()
```