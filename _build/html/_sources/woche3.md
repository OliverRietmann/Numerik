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

# Woche 3: Taylor Approximation

Lernziele:

Sei eine Stelle $x_0$ und eine Funktion $f(x)$ gegeben.

1. Ich kann in Python die Linearisierung von $f$ an der Stelle $x_0$ berechnen.
2. Ich kann in Python das Taylor-Polynom vom Grad $n$ von $f$ an der Stelle $x_0$ berechnen.

## Linearisierung

Berechne die Linearisierung $t_1(x)$ der Funktion $f(x)=\cos(x)$ an der Stelle $x_0=1$.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

x0 = 1.0
f = lambda x: np.cos(x)
df = lambda x: -np.sin(x)
t1 = lambda x: f(x0) + (x - x0) * df(x0)
x = np.linspace(-np.pi, np.pi, 100)

plt.figure()
plt.plot(x, t1(x), label='$t_1(x)$')
plt.plot(x, f(x), '--', label="$f(x)$")
plt.legend()
plt.show()
```

## Taylor Polynome

Nun approximieren wir die Funktion $f(x)=\cos(x)$ an der Stelle $x_0=1$ mit einem Taylor Polynom $t_n(x)$ vom Grad $n$, also

$$
t_n(x)=\sum\limits_{k=0}^n c_k\cdot (x-x_0)^k,\qquad c_k=\frac{f^{(k)}}{k!}.
$$


```{code-cell} ipython3
import math
import numpy as np
import matplotlib.pyplot as plt

def taylor_coefficients(x0, fk_list):
    derivatives = np.array([fk(x0) for fk in fk_list])
    n = len(fk_list) - 1
    factorials = np.array([math.factorial(k) for k in range(n + 1)])
    return derivatives / factorials

def taylor_evaluation(x, x0, c):
    n = len(fk_list) - 1
    return sum([c[k] * (x - x0)**k for k in range(n + 1)])

f0 = lambda x: np.cos(x)
f1 = lambda x: -np.sin(x)
f2 = lambda x: -np.cos(x)
f3 = lambda x: np.sin(x)

x0 = 1.0
fk_list = [f0, f1, f2, f3]
c = taylor_coefficients(x0, fk_list)
print("Taylor-Koeffizienten: ", c)

x = np.linspace(-np.pi, np.pi, 100)

plt.figure()
plt.plot(x, taylor_evaluation(x, x0, c), label='$t_3(x)$')
plt.plot(x, f0(x), '--', label="$f(x)$")
plt.legend()
plt.show()
```
