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

# Woche 2: Grundlagen der Numerik mit Python

Lernziele:

1. Ich kann `if` / `else` Blöcke nutzen.
2. Ich kann `for` und `while` Schleifen implementieren.

## Endliche Arithmetik

Die Ableitung einer stetig differenzierbaren Funktion $$f$$ an der Stelle $x_0$ ist definiert als

$$ f^\prime(x0)= \lim\limits_{h\rightarrow 0} \frac{f(x_0+h)-f(x_0)}{h} $$.

Für $$ f(x)=\tfrac{1}{2}x^2 $$ erhalten wir zum Beispiel $f^\prime(x)=x$.
Folgender Code appriximiert also für `h` nahe bei Null die Ableitung an der Stelle `x0`.
```{code-cell}
import numpy as np
from matplotlib import pyplot as plt

def derivative(g, x0, h):
    return (g(x0 + h) - g(x0)) / h

def f(x):
    return 0.5 * x**2

h_values = np.array([2**(-k) for k in range(10, 30)])
df_values = derivative(f, 1.0, h_values)
error = np.abs(1.0 - df_values)

print(error)
```

Archmides entwickelte einen Algorithmus zur Berechnung der Kreiszahl $$\pi$$.
Dabei berechnet er die Fläche des Einheitskreises (Radius 1), welche genau $$\pi$$ entspricht.
Die Fläche wird durch

```{image} ./images/archimedes.jpg
:alt: archimedes
:class: fig
:width: 400px
:align: center
```

## Kontrollstrukturen

Letzte Woche haben wir folgendes Programm geschrieben.
Es gibt aus, ob die Zahl `n` gerade oer ungerade ist.
Hier bezeichnet `%` den Modulo-Operator (Rest aus Division).
```{code-cell}
n = 3

if n % 2 == 1:
  print(n, "ist gerade")
else:
  print(n, "ist ungerade")
```

:::{admonition} Aufgabe
Ergänzen Sie folgenden Python Code, so dass er die Binärdarstellung der natürlichen Zahl `n` ausgiebt.
:::
```{code-cell}
# Zum Beispiel 6 --> 110 oder 26 --> 11010
n = 6

while n > 0:
  if n % 2 == 0:
    print(0)
  else:
    print(1)
  n = n // 2
```