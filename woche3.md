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