#!/usr/bin/env python
# coding: utf-8

# # Woche 1: Einführung in Python
# 
# Lernziele:
# 
# 1. Ich kann mit der Klasse `numpy.array` Vektoren und Matrizen erstellen diese manipulieren.
# 2. Ich weiss, was die Operatoren `+,-,*,/,**` mit einem `numpy.array` machen.
# 3. Ich kenne den Unterschied zwischen `numpy.sqrt` und `math.sqrt` (analog für weitere Funktionen).
# 4. Ich kann mit dem Package `matplotlib` die Funktion `numpy.sin` plotten.
# 5. Ich kann in Python eigene Funktionen definieren und diese ausführen.
# 
# ## Variablen, Zeichenketten (Strings), Listen und Schleifen
# 
# Was ist der Output des folgenden Codes?

# In[1]:


x = 3
print(x)
print("x")

var = "I am a string."
print(var)


# In[2]:


l = [1, 2, 3]
print(l)
print(l[1])
print(l[-1])

l = [1, "two", 3]
print(l)
for n in l:
  print(n)
  print("Within the loop.")
print("Outside of the loop.")


# In[3]:


for i in range(3):
  if i % 2 == 0:
    print(i)


# ## Python als Taschenrechner
# 
# Was ist der Output des folgenden Codes?

# In[4]:


print(2 + 3)
print(2 * 3)
print(2**3)


# Whether you write your book's content in Jupyter Notebooks (`.ipynb`) or
# in regular markdown files (`.md`), you'll write in the same flavor of markdown
# called **MyST Markdown**.
# This is a simple file to help you get started and show off some syntax.
# 
# ## What is MyST?
# 
# MyST stands for "Markedly Structured Text". It
# is a slight variation on a flavor of markdown called "CommonMark" markdown,
# with small syntax extensions to allow you to write **roles** and **directives**
# in the Sphinx ecosystem.
# 
# For more about MyST, see [the MyST Markdown Overview](https://jupyterbook.org/content/myst.html).
# 
# ## Sample Roles and Directives
# 
# Roles and directives are two of the most powerful tools in Jupyter Book. They
# are kind of like functions, but written in a markup language. They both
# serve a similar purpose, but **roles are written in one line**, whereas
# **directives span many lines**. They both accept different kinds of inputs,
# and what they do with those inputs depends on the specific role or directive
# that is being called.
# 
# Here is a "note" directive:
# 
# ```{note}
# Here is a note
# ```
# 
# It will be rendered in a special box when you build your book.
# 
# Here is an inline directive to refer to a document: {doc}`woche2`.
# 
# 
# ## Citations
# 
# You can also cite references that are stored in a `bibtex` file. For example,
# the following syntax: `` {cite}`holdgraf_evidence_2014` `` will render like
# this: {cite}`holdgraf_evidence_2014`.
# 
# Moreover, you can insert a bibliography into your page with this syntax:
# The `{bibliography}` directive must be used for all the `{cite}` roles to
# render properly.
# For example, if the references for your book are stored in `references.bib`,
# then the bibliography is inserted with:
# 
# ```{bibliography}
# ```
# 
# ## Learn more
# 
# This is just a simple starter to get you started.
# You can learn a lot more at [jupyterbook.org](https://jupyterbook.org).
