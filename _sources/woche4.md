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

# Woche 4: Nullstellen nichtlinearer Funktionen

Lernziele:

1. Ich kann eine nichtlineare Gleichung in eine Nullstellensuche umschreiben.
2. Ich kann die Bisektion in Python implementieren.
3. Ich kann das Newton Verfahren in Python implementieren.
4. Ich kenne die Vor- und Nachteile dieser Methoden.

## Bisektion

Anführungszeichen `"..."` definieren eine Zeichenkette (engl. string).
```{code-cell} ipython3
def bisection(f, a, b, tol):
    assert(f(a) * f(b) < 0)
    while abs(b-a) > tol:
        m = (a + b) / 2
        fm = f(m)
        if fm == 0.0:
            return m, m
        elif fm * f(b) < 0:
            a = m
        else:
            b = m
    return [a, b]

f = lambda x: x**2 - 3
a = 1
b = 2

# sqrt(3) = 1.7320508075688772
print(bisection(f, a, b, 1.0e-3))
```

## Newton Verfahren

Das Heronsche Näherungsverfahren approximiert die Wurzel $\sqrt{a}$ einer positiven Zahl $a$, oder äquivalent die Nullstelle von $f(x)=x^2-a$.
Es ist definiert durch die Folge

$$
x_{k+1}=\frac{1}{2}\bigg(x_k+\frac{a}{x_k}\bigg)
$$

und diese konvergiert dann gegen $\sqrt{a}$.

```{code-cell} ipython3
def heron(a, x, tol):
    while abs(x**2 - a) > tol:
        x = 0.5 * (x + a / x)
    return x

a = 3.0
x = 2.0

# sqrt(3) = 1.7320508075688772
print(newton(x, a, 1.0e-3))
```

Das Newton Verfahren für eine stetig differenzierbare Funktion $f:\mathbb R\rightarrow\mathbb R$ ist definiert durch die Folge

$$
x_{k+1}=x_k-\frac{f(x_k)}{f^\prime(x_k)}
$$

mit einem geeigneten Startwert $x_0$.
Dieser sollte nahe bei der tatsächlichen Nullstelle $x^\ast$ liegen.

```{code-cell} ipython3
def newton(f, df, x, tol):
    while abs(f(x)) > tol:
        x = x - f(x) / df(x)
    return x

f = lambda x: x**2 - 3
df = lambda x: 2 * x
x = 2.0

# sqrt(3) = 1.7320508075688772
print(newton(f, df, x, 1.0e-3))
```