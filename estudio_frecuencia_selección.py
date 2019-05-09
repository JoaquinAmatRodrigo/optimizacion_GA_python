#%%
import numpy as np
import random
import warnings
import copy
import pandas as pd
import time
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime
from operator import itemgetter 
#%%
array_fitness = np.array([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
                          6, 5, 4, 3, 2, 1])
probabilidad_seleccion = array_fitness / np.sum(array_fitness)
ind_seleccionado = np.random.choice(
                    a    = np.arange(len(array_fitness)),
                    size = 500,
                    p    = list(probabilidad_seleccion),
                    replace = True
                )
    
pd.value_counts(pd.Series(ind_seleccionado)) \
    .plot(kind="bar", ylim=(0,200), title = "Ruleta")

#%%
array_fitness = np.array([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
                          6, 5, 4, 3, 2, 1])
order = np.flip(np.argsort(a=array_fitness) + 1)
ranks = np.argsort(order) + 1
probabilidad_seleccion = 1 / ranks
probabilidad_seleccion = probabilidad_seleccion / np.sum(probabilidad_seleccion)
ind_seleccionado = np.random.choice(
                    a    = np.arange(len(array_fitness)),
                    size = 500,
                    p    = list(probabilidad_seleccion),
                    replace = True
                   )
ind_seleccionado
pd.value_counts(pd.Series(ind_seleccionado)) \
    .plot(kind="bar", ylim=(0,200), title = "Rank")

#%%
array_fitness = np.array([20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7,
                          6, 5, 4, 3, 2, 1])
indices_seleccionados = np.repeat(None,500)
for i in np.arange(500):
    # Se seleccionan aleatoriamente dos parejas de individuos.
    candidatos_a = np.random.choice(
                    a       = np.arange(len(array_fitness)),
                    size    = 2,
                    replace = False
                    )
    candidatos_b = np.random.choice(
                    a       = len(array_fitness),
                    size    = 2,
                    replace = False
                    )
    # De cada pareja se selecciona el de mayor fitness.
    if array_fitness[candidatos_a[0]] > array_fitness[candidatos_a[1]]:
        ganador_a = candidatos_a[0]
    else:
        ganador_a = candidatos_a[1]

    if array_fitness[candidatos_b[0]] > array_fitness[candidatos_b[1]]:
        ganador_b = candidatos_b[0]
    else:
        ganador_b = candidatos_b[1]

    # Se comparan los dos ganadores de cada pareja.
    if array_fitness[ganador_a] > array_fitness[ganador_b]:
        ind_seleccionado = ganador_a
    else:
        ind_seleccionado = ganador_b

    indices_seleccionados[i] = ind_seleccionado

pd.value_counts(pd.Series(indices_seleccionados)) \
    .plot(kind="bar", ylim=(0,200), title = "Tournament")

#%%
