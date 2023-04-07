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

## Lineare Gleichungssysteme

Wir betrachten das lineare Gleichungssystem (LGS) $Ax=b$, wobei

$$
A:=
\begin{pmatrix}
    1 & -2 & -1 \\
    2 & -1 & 1 \\
    3 & -6 & -5
\end{pmatrix},\qquad
b:=
\begin{pmatrix}
    3 \\
    0 \\
    3
\end{pmatrix}.
$$

Hier ist $A$ eine Matrix und $b$ ein Vektor.
Beide sind gegeben.
Der Vektor $x$ ist die gesuchte Grösse.
Wir betrachten nur den Spezialfall wo $A$ eine **quadratische** Matrix ist.

Wir berechnen die $LU$-Zerlegung (ohne Zeilenvertauschung).

```{code-cell} ipython3
import numpy as np

def LUdecomposition(A):
    n = len(A)
    L = np.eye(n)
    U = A
    for k in range(n):
        L[k+1:n, k] = U[k+1:n, k] / U[k, k]
        for j in range(k + 1, n):
            U[j, :] = U[j, :] - L[j, k] * U[k, :]
    return L, U

A = np.array([[1.0, -2.0, -1.0],
              [2.0, -1.0,  1.0],
              [3.0, -6.0, -5.0]])

L, U = LUdecomposition(A)

print(L)
print(U)
print(L @ U)
```

Wir haben die $LU$-Zerlegung von $A$ bereits berechnet:

$$
\begin{pmatrix}
    1 & -2 & -1 \\
    2 & -1 & 1 \\
    3 & -6 & -5
\end{pmatrix}
=
\begin{pmatrix}
    1 & 0 & 0 \\
    2 & 1 & 0 \\
    3 & 0 & 0
\end{pmatrix}
\cdot
\begin{pmatrix}
    1 & -2 & -1 \\
    0 & 3 & 3 \\
    0 & 0 & -2
\end{pmatrix}
$$

Nun können wir das LGS lösen.

```{code-cell} ipython3
import numpy as np

def forward(L, b):
    y = np.zeros_like(b)
    for i in range(len(b)):
        y[i] = (b[i] - np.dot(L[i], y)) / L[i, i]
    return y

def backward(U, y):
    x = np.zeros_like(y)
    for i in reversed(range(len(y))):
        x[i] = (y[i] - np.dot(U[i], x)) / U[i, i]
    return x

L = np.array([[1.0, 0.0, 0.0],
              [2.0, 1.0, 0.0],
              [3.0, 0.0, 1.0]])
U = np.array([[1.0, -2.0, -1.0],
              [0.0,  3.0,  3.0],
              [0.0,  0.0, -2.0]])
b = np.array([3.0, 0.0, 3.0])

y = forward(L, b)
x = backward(U, y)

print("y =", y) # [ 3. -6. -6.]
print("x =", x) # [-4. -5.  3.]
```

## Numerische Instabilität

Wir betrachten ein neues LGS $Ax=b$, wobei

$$
A:=
\begin{pmatrix}
    a & 1 & 0 \\
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

mit der sehr kleinen Zahl $a=10^{-20}$.
Hier wäre als fast eine Zeilenvertauschung nötig (falls $a=0$ wäre sie tatsächlich nötig).
Die Lösung des LGS $Ax=b$ lautet

$$
x=
\begin{pmatrix}
    0 \\
    1 \\
    1000
\end{pmatrix}
$$

Das entsprechende LGS lösen wir zuerst mit unserer Implementierung der $LU$-Zerlegung.

```{code-cell} ipython3
import numpy as np

# np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.0f}'.format}) 

a = 1.0e-20
A = np.array([[a, 1.0, 0.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 0.0]])
b = np.array([1.0, 1000.0, 1.0])

L, U = LUdecomposition(A)
y = forward(L, b)
x = backward(U, y)

print("x =", x) # [0, 1, 1000]
```

:::{admonition} Aufgabe
Wo liegt das Problem und wie können wir es beheben?
:::

Nun lösen wir das LGS mit `numpy.linalg.solve(...)`.

```{code-cell} ipython3
import numpy as np

a = 1.0e-20
A = np.array([[a, 1.0, 0.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 0.0]])
b = np.array([1.0, 1000.0, 1.0])

x = np.linalg.solve(A, b)

print("x =", x) # [0, 1, 1000]
```

## Konditionszahl

Die Konditionszahl einer Matrix $A$ ist definiert als

$$
\kappa(A):=\lVert A\rVert\cdot\lVert A^{-1}\rVert
$$

und kann in Python berechnet werden mit `numpy.linalg.cond(...)`.

```{code-cell} ipython3
import numpy as np

a = 1.0e-20
A = np.array([[a, 1.0, 0.0],
              [1.0, 0.0, 1.0],
              [1.0, 1.0, 0.0]])
P = np.array([[0.0, 1.0, 0.0],
              [1.0, 0.0, 0.0],
              [0.0, 0.0, 1.0]])

print(np.linalg.cond(A))
print(np.linalg.cond(P @ A))
```