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

# Woche 7: Gauss-Algorithmus

Lernziele:



## Linear Gleichungssysteme

Ein lineares Gleichungssystem (LGS) ist eine Gleichung der Form

$$
Ax=b.
$$

Hier ist $A$ eine Matrix und $b$ ein Vektor.
Beide sind gegeben.
Der Vektor $x$ ist die gesuchte Grösse.
Wir betrachten hier nur den Spezialfall wo $A$ eine **quadratische** Matrix ist.
Hier ist ein Beispiel

$$
A:=
\begin{pmatrix}
    0 & 1 & 0 \\
    1 & 0 & 1 \\
    1 & 1 & 0
\end{pmatrix},\qquad
b:=
\begin{pmatrix}
    1 \\
    1000 \\
    1
\end{pmatrix}
$$

Das entsprechende LGS lösen wir in Python mit `numpy.linalg.solve(...)`.

```{code-cell} ipython3
import numpy as np

A = np.array([[1.0e-20, 1.0, 0.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 0.0]])
b = np.array([1.0, 1000.0, 1.0])

x = np.linalg.solve(A, b)

print(x)
print(np.dot(A, x))
```

:::{admonition} Aufgabe
test
:::