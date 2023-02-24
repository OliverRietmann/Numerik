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

# Woche 1: Einführung in Python

Lernziele:

1. Ich kann mit der Klasse `numpy.array` Vektoren und Matrizen erstellen diese manipulieren.
2. Ich weiss, was die Operatoren `+,-,*,/,**` mit einem `numpy.array` machen.
3. Ich kenne den Unterschied zwischen `numpy.sqrt` und `math.sqrt` (analog für weitere Funktionen).
4. Ich kann mit dem Package `matplotlib` die Funktion `numpy.sin` plotten.
5. Ich kann in Python eigene Funktionen definieren und diese ausführen.

## Variablen, Zeichenketten (Strings), Listen und Schleifen

Anführungszeichen `"..."` definieren eine Zeichenkette (engl. string).
```{code-cell} ipython3
x = 3
var = "I am a string."
print(x, "x", var)
```

Eckige Klammern `[...]` definieren eine Liste.
Auf die einzelnen Elemente einer Liste greifen wir mit `[i]` zu, wobei `i` eine ganze Zahl bezeichnet.
```{code-cell}
l = [1, 2, 3]
print(l[1])
```

Mit einer `for` Schleife (engl. loop) können wir über Listen iterieren.
Der eingerückte Code wir dabei wiederholt.
```{code-cell}
l = [1, "two", 3]
print(l)
for n in l:
  print(n)
```

:::{admonition} Aufgabe
Ergänzen Sie folgenden Code mit einer `if` Anweisung, so dass er angibt, ob die Variable `n` gerade oder ungerade ist.
:::
```{code-cell}
n = 3
# Ihr Code kommt hier hin:
# ...
# ...
```


## Python als Taschenrechner

:::{admonition} Aufgabe
Was ist der Output des folgenden Codes?
:::
```{code-cell}
print(2 + 3)
print(2 * 3)
print(2 / 3)
print(2**3)
print(2^3)
```

Das Modul (engl. module) `math` enthält diverse Funktionen aus der Mathematik.
```{code-cell}
import math

print(math.sqrt(45))
print(math.pow(23, 1 / 5))
```

## Funktionen

Wir können auch unsere eigenen Funktionen definieren.
```{code-cell}
def log17(x):
  y = math.log(x) / math.log(17)
  return y
```
:::{admonition} Aufgabe
Welchen Wert haben `log17(17)` und `log17(17**2)`?
:::

Eine Funktion kann auch mehrere Argumente haben.
:::{admonition} Aufgabe
Folgende Funktion soll zwei gleich lange Listen elementweise addieren. Korrigieren Sie die Funktion.
:::
```{code-cell}
l1 = [1, 2, 3]
l2 = [4, 5, 6]

def sum_lists(v, w):
  return v + w

print(sum_lists(l1, l2))
```

Eine Funktion kann auch mehrere Argumente haben.
:::{admonition} Aufgabe
Folgende Funktion soll eine Skalar-Vektor Multiplikation machen. Korrigieren Sie die Funktion.
:::
```{code-cell}
l = [1, 2, 3]

def skalar_mult(s, v):
  return s * v

print(skalar_mult(3, l))
```

## Vektoren und Matrizen

Das Packet (engl. package) `numpy` enthält die wichtigsten Funktionen aus der linearen Algebra.
Es kann Listen in `numpy.arrays` umwandeln.
Diese verhalten sich wie Vektoren und Matrizen.
```{code-cell}
import numpy as np

l = [1, 2, 3]
x = np.array(l)
y = np.array([4, 5, 6])

print(x + y)
print(3 * x)
```