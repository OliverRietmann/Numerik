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
4. Ich kann mit dem package `matplotlib` die Funktion `numpy.sin` plotten (analog für weitere Funktionen).
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
def sum_lists(v, w):
  return v + w

l1 = [1, 2, 3]
l2 = [4, 5, 6]
print(sum_lists(l1, l2))
```

:::{admonition} Aufgabe
Folgende Funktion soll eine Skalar-Vektor Multiplikation machen.
Korrigieren Sie die Funktion.
:::
```{code-cell}
def skalar_mult(s, v):
  return s * v

l = [1, 2, 3]
print(skalar_mult(3, l))
```

## Vektoren und Matrizen

Das Paket (engl. package) `numpy` enthält die wichtigsten Funktionen aus der linearen Algebra.
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

Im Gegensatz zu `math` agieren die `numpy` Funktionen elementweise.
```{code-cell}
import numpy as np

x = np.array([1, 2, 3])
print(np.sqrt(x))
```

Das Skalarprodukt zweier Vektoren berechnet man mit `numpy.dot`.
```{code-cell}
import numpy as np

v = np.array([1, 2, 3])
w = np.array([0, 0, 1])
print(np.dot(v, w))
```

Matrizen entstehen aus Listen von Listen.
In diesem Fall berechnet `numpy.dot` das Matrix-Matrix Produkt.
```{code-cell}
import numpy as np

A = np.array([[16,  3],
              [ 5, 10],
              [ 9,  6]])

B = np.array([[1, 2, 3],
              [4, 6, 6]])

print(np.dot(A, B))
# Gleich wie A.dot(B) und A @ B
# Probieren Sie mal np.shape(A)
```

Es gibt effiziente Funktionen um spezielle Matrizen zu erstellen.
| Funktion         | Beschreibung   |
|------------------|:---------------|
| `eye`            | Einheitsmatrix |
| `zeros`          | Nullmatrix     |
| `ones`           | Alles Einsen   |
| `diag`           | Diagonalmatrix |
| `random.random`  | Alles Einsen   |
```{code-cell}
import numpy as np

R = np.random.random((2,3))
print(R)
```

Mit `[...]` kann man auf die Einträge von Matrizen und Vektoren zugreifen.
```{code-cell}
import numpy as np

v = np.array([1, 2, 3])
print(v[1])

A = np.array([[16,  3],
              [ 5, 10],
              [ 9,  6]])
print(A[2, 1])
```

Das `numpy.array` übernimmt den Datentyp der Zahlen in der Liste im Argument.
:::{admonition} Aufgabe
Was ist der Output des folgenden Codes? Erklären Sie.
:::
```{code-cell}
import numpy as np

x = np.array([1])
x[0] = x[0] / 2
print(x[0])
```
:::{admonition} Aufgabe
1. Berechnen Sie den Sinus von 30 Grad in Python.
2. Überlegen Sie sich Varianten zur Definition der Matrix

$$
\begin{pmatrix}
  0 & 1 & 1 & 1 & 1 \\
  1 & 0 & 1 & 1 & 1 \\
  1 & 1 & 0 & 1 & 1 \\
  1 & 1 & 1 & 0 & 1 \\
  1 & 1 & 1 & 1 & 0 \\
\end{pmatrix}.
$$
:::
```{code-cell}
import numpy as np

# Ihr Code kommt hier hin.
```

## Graphische Darstellungen

Um Funktionen zu plotten müssen wir sie an vielen verschiedenen Positionen auf der x-Achse auswerten.
Dazu gibt es die Funktion `numpy.linspace`.
```{code-cell}
import numpy as np

x = np.linspace(1, 5, 9)
print(x)
```
:::{admonition} Aufgabe
Seien `start < stop` und sei `num` eine ganze Zahl.
Beschreiben Sie in Worten, was der Aufruf `numpy.linspace(start, stop, num)` macht.
:::

Zum Plotten verwenden wir das Modul `matplotlib.pyplot`.
```{code-cell}
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

plt.figure()
plt.plot(x, y, color="red")
plt.title("Plot von sin(x)")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.show()
```
:::{admonition} Aufgabe
Ändern Sie obigen Code, so dass er das Polynom

$$ p(x)=x^2 + x + 17 $$

plottet.
:::

Oft will man mehrere Funktionen gleichzeitig plotten.
:::{admonition} Aufgabe
Ergänzen Sie folgenden Code, so dass er die drei Funktionen im selben Koordinatensystem plottet.
:::
```{code-cell}
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 20, 100)
y1 = np.sin(x) / np.sqrt(x + 1)
y2 = np.sin(x / 2) / np.sqrt(x + 1)
y3 = np.sin(x / 3) / np.sqrt(x + 1)

plt.figure()
plt.xlabel("x")
# Ihr Code kommt hier hin
# ...
plt.show()
```

Oder man mach separate Plots.
```{code-cell}
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)

fig, ax = plt.subplots(2)
fig.suptitle("Vertically stacked subplots")
ax[0].plot(x,  y, color="red" )
ax[1].plot(x, -y, color="blue")
plt.show()
```
:::{admonition} Aufgabe
Was müssen Sie ändern, damit die Plots nebeneinander stehen?
:::


## Weitere Aufgaben

:::{admonition} Aufgabe 1
Schreiben Sie ein Skript, das die Funktionen $x = \cos(t)$ und $y = \sin(t)$ mit $t$ im Intervall $[0, 2\pi]$ auswertet und mittels `matplotlib.pyplot.plot(x, y)` darstellt.
(Hilfestellung: Sie brauchen geeignete Stützstellen $t$ im Intervall.)
:::

:::{admonition} Aufgabe 2
Modifizieren Sie das vorangegangene Programm zu einer Python-Funktion, die als Argumente die linke und rechte Intervallgrenze $a$ und $b$ des $t$-Intervalls entgegennimmt.
:::

:::{admonition} Aufgabe 3
Schreiben Sie eine Python-Funktion, die die Funktionen

$$ x = \sin(At+a) \quad\text{und}\quad y = \sin(Bt+b) $$

für $t$ im Intervall $[0, 2\pi]$ auswertet und mittels `matplotlib.pyplot.plot(x, y)` darstellt.
Dabei sollen $A, a$ und $B, b$ die Argumente der Funktion sein.
:::

:::{admonition} Aufgabe 4
Experimentieren Sie mit folgenden Werten:

$$ A = 2, a = \pi / 4, B = 1, b = 0 $$
$$ A = 3, a = \pi / 2, B = 1, b = 0 $$
$$ A = 1, a = 0, B = 1, b = 0 $$

Variieren Sie dann weiter $A, a$ und $B, b$.
Was stellen Sie fest? Beschreiben Sie ihre Beobachtungen in Worten.
Welche Bedeutung haben $a, b$ sowie $A, B$? Recherchieren Sie asserdem im Internet das Thema ”Lissajous Figuren”.
:::
