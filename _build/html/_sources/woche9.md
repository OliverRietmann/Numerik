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

# Woche 9: Interpolation

Lernziele:

1. Ich kann aus der Interpolationsbedinung ein LGS für die Koeffizienten bestimmen.
2. Ich kann mit `numpy.linalg.solve(...)` ein LGS lösen.
3. Ich kann eine Polynom-Interpolation mit den Funktionen `numpy.polyfit(...)` und `numpy.polyval(...)` ausführen.

## Polynom-Interpolation

Wir wollen ein das Interpolationspolynom $p_n(x)$ durch die Punkte $(x_i,y_i),i=0,\ldots,n$ berechnen, wobei

$$
p_n(x)=a_0+a_1x+a_2x^2+\cdots+a_nx^n.
$$

Die Interpolationsbedingung liefert das LGS

$$
\begin{pmatrix}
    1 & x_0 & x_0^2 & \cdots & x_0^n \\
    1 & x_1 & x_1^2 & \cdots & x_1^n \\
    \vdots & \vdots & \vdots & \vdots & \vdots \\
    1 & x_n & x_n^2 & \cdots & x_n^n \\
\end{pmatrix}
\begin{pmatrix}
    a_0 \\
    a_1 \\
    a_2 \\
    \vdots \\
    a_n
\end{pmatrix}
=
\begin{pmatrix}
    y_0 \\
    y_1 \\
    y_2 \\
    \vdots \\
    y_n
\end{pmatrix}.
$$

Wir lösen dieses mit `numpy.linalg.solve(...)`.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
y = np.array([1.0, 1.0, 0.0, 0.0, 3.0])

V = np.vander(x, increasing=True)
a = np.linalg.solve(V, y)
p = lambda x: sum([a[i] * x**i for i in range(len(a))])

x_values = np.linspace(0.0, 2.0, 100)
plt.figure()
plt.plot(x, y, 'bo')
plt.plot(x_values, p(x_values), 'r-')
plt.show()
```

Alternativ kann man auch `numpy.polyfit(...)` und `numpy.polyval(...)` verwenden.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
y = np.array([1.0, 1.0, 0.0, 0.0, 3.0])

p = np.polyfit(x, y, len(x))

x_values = np.linspace(0.0, 2.0, 100)
plt.figure()
plt.plot(x, y, 'bo')
plt.plot(x_values, numpy.polyval(p, x_values), 'r-')
plt.show()
```

## Lagrange Polynome

Seinen wieder Punkte $(x_i,y_i),i=0,\ldots,n$ gegeben.
Unser Interpolationspolynom ist nun von der Form

$$
p_n(x)=y_0\ell_0(x)+y_1\ell_1(x)+\cdots+y_n\ell_n(x)
$$

mit den Lagrange Polynomen

$$
\ell_i(x)=\prod\limits_{k\neq i}\frac{x-x_k}{x_i-x_k}.
$$

Der folgende Plot visualisiert die definierende Eigenschaft

$$
\ell_i(x_k)=
\begin{cases}
0,\ i\neq k\\
1,\ i=k
\end{cases}
$$

der Lagrange Polynome.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

x = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
y = np.array([1.0, 1.0, 0.0, 0.0, 3.0])

def Lagrange_factory(x, i):
    xi = x[i]
    x_without_i = np.delete(x, [i])
    return lambda z: np.prod([(z - xk) / (xi - xk) for xk in x_without_i], axis=0)

n = len(x)
f = [Lagrange_factory(x, i) for i in range(n)]

x_values = np.linspace(0.0, 2.0, 100)
plt.figure()
plt.plot(x, y, 'bo')
plt.plot(x_values, sum(y[i] * f[i](x_values) for i in range(n)), 'k-', label=r"$p_{0}(x)$".format(n + 1))
for i in range(n):
	plt.plot(x_values, f[i](x_values), '--', label=r"$f_{0}(x)$".format(i))
plt.legend()
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

## Splines

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
\end{pmatrix},\qquad
y:=
\begin{pmatrix}
    y_1 \\
    y_2 \\
    \vdots \\
    y_n
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