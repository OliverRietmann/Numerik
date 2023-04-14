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

1. Ich zu einer gegebenen Linearkombination von Funktionen die zugehörige Normalengleichung aufstellen.
2. Ich kann die Normalengleichung mit `numpy.linalg.solve(...)` lösen.
3. Ich kann die Funktionen `numpy.polyfit` und `numpy.polyval` anwenden.

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
    1 & x_n
\end{pmatrix},\qquad
y:=
\begin{pmatrix}
    y_1 \\
    y_2 \\
    \vdots \\
    y_n
\end{pmatrix}.
$$

Wir lösen die Normalengleichung mit `numpy.linalg.solve(...)`.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

n = 30
x = np.linspace(-2.5, 2.5, n)
noise = np.random.rand(n) - 0.5
y = 2.0 * x + 3.0 + noise

A = np.column_stack((np.ones(n), x))
AT = np.transpose(A)
b, m = np.linalg.solve(AT @ A, np.dot(AT, y))
print(m, b)

plt.figure()
plt.plot(x, y, 'bo')
plt.plot(x, m * x + b, 'r-')
plt.show()
```

## Allgemeine Polynome

Nun wollen wir ein Polynom von Grad 3 fitten.
Die Normalengleichung für den Koeffizientenvektor $p$ dieses Polynoms lautet

$$
A^TA\cdot
\begin{pmatrix}
    b \\
    m
\end{pmatrix}
=A^Ty,\qquad
A:=
\begin{pmatrix}
    1 & x_1 & x_1^2 & x_1^3 \\
    1 & x_2 & x_2^2 & x_2^3 \\
    \vdots & \vdots & \vdots & \vdots \\
    1 & x_n & x_n^2 & x_n^3 \\
\end{pmatrix}.
$$

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

n = 30
x = np.linspace(-2.5, 2.5, n)
noise = 0.5 * np.random.rand(n) - 0.25
y = np.tanh(x) + noise

A = np.column_stack((np.ones(n), x, x**2, x**3))
AT = np.transpose(A)
p = np.linalg.solve(AT @ A, np.dot(AT, y))
print(p)

plt.figure()
plt.plot(x, y, 'bo')
plt.plot(x, np.dot(A, p), 'r-')
plt.show()
```

Alternativ können auch die Funktionen `numpy.polyfit(...)` und `numpy.polyval(...)` verwendet werden.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

n = 30
x = np.linspace(-2.5, 2.5, n)
noise = 0.5 * np.random.rand(n) - 0.25
y = np.tanh(x) + noise

degree = 3
p = np.polyfit(x, y, degree)
print(p)

plt.figure()
plt.plot(x, y, 'bo')
plt.plot(x, np.polyval(p, x), 'r-')
plt.show()
```

## Beliebige Funktionen

Nun wollen wir eine Funktion der Form

$$
f(x)=p_0\cdot\sin(x)+p_1\cdot x
$$

fitten.
Die Normalengleichung für den Koeffizientenvektor $p=(p_0,p_1)^T$ lautet

$$
A^TA\cdot p=A^Ty,\qquad
A:=
\begin{pmatrix}
    \sin(x_1) & x_1 \\
    \sin(x_2) & x_2 \\
    \vdots & \vdots \\
    \sin(x_n) & x_n
\end{pmatrix}.
$$

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

n = 30
x = np.linspace(-5.0, 5.0, n)
noise = 0.5 * np.random.rand(n) - 0.25
y = np.sin(x) + 0.5 * x + noise

A = np.column_stack((np.sin(x), x))
AT = np.transpose(A)
p = np.linalg.solve(AT @ A, np.dot(AT, y))
print(p)

plt.figure()
plt.plot(x, y, 'bo')
plt.plot(x, np.dot(A, p), 'r-')
plt.show()
```