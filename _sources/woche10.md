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

## Trapezregel

Wir approximieren das Integral

$$
\int_^\pi\sin(x)dx
$$

mit einer zusammengesetzte Trapezregel.
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