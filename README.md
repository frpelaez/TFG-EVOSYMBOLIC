En este repositorio e encuentra el código asociado a mi Trabajo de Fin de Grado. He terminado el Grado de Matemáticas en la Universidad Complutense de Madrid y aquí explico brevemente el funcionamiento del proyecto.

He atacado el problema de la regresión simbólica, es decir, encontrar expresiones matemáticas que se ajusten a un conjunto de datos de entrada. Lo he hecho por medio de técnicas de programación genética. El funcionamiento general es el siguiente: se genera una población inicial de individuos (árboles sintácticos que representan las expresiones) y se actualiza la población un número determinado de iteraciones (generaciones). La actualización consiste en tomar a los individuos de la población y aplicarles los operadores genéticos de cruzamiento y mutación. Estos recombinan el material genético en la población y después se construye la nueva generación seleccionando algunos de los mejores individuos producidos. Una vez concluido el proceso, se devuelve el mejor individuo de la última generación.

He aquí un pequeño ejemplo de funcionamiento:

```Python
import numpy as np
import sympy as sp 
from tfg_evosymbolic.evosym_run import evosym
from tfg_evosymbolic.symbolic_expression import expr_from_tree

# Intervalo de tiempo que utilizamos [-2, 2]
t = np.linspace(-2.0, 2.0, 50)
# Simulamos los datos como y = e^-t * cos(t^2)
y = np.exp(-t) * np.cos(t**2) 
# Definimos el conjunto de datos
datos = (t, y)
# Definimos una variable symbolica
x = sp.Symbol("x")
varales = [x]
# Definimos el conjunto de operadores/aridades a utilizar
operadores = {"+":2, "-":2, "*":2, "exp":1, "sin":1, "cos":1}
# Definimos el rango para las contsantes numéricas
constantes = [-5.0, 5.5]

# Realizamos una ejecución del algoritmo genético
solucion, _ = evosym(
	datos,
	50,        # Tamaño de la población
	100,       # Número de generaciones
	variables,
	operadores,
	constants_range=constantes
)

# Mostramos la expresión encontrada
print(expr_from_tree(solucion.tree)) # e.j. sin(x*x + 1.57) * exp(-x) 
```

En el proyecto nos hemos centrado bastante en cómo, en ciertas ocasiones, añadir el criterio de la derivada puede resultar beneficioso. Esto consiste en juzgar a los individuos no únicamente por cuánto se asemejan a los datos de entrada sino también por cuán parecidas son las expresiones derivadas de estos individuos a una determinada representación discreta de la derivada de los datos. La forma de incluirlo en el código es sencilla, la mostramos a continuación:

```Python
import numpy as np
import sympy as sp 
from tfg_evosymbolic.evosym_run import evosym
from tfg_evosymbolic.symbolic_expression import expr_from_tree
from tfg_evosymbolic.utils import calculate_deriv

# Intervalo de tiempo que utilizamos [-2, 2]
t = np.linspace(-2.0, 2.0, 50)
# Simulamos los datos como y = e^-t * cos(t^2)
y = np.exp(-t) * np.cos(t**2) 
# Definimos el conjunto de datos
datos = (t, y)
# Calculamos la derivada de los datos mediante métodos numéricos
datos_deriv = calculate_deriv(datos)
# Definimos una variable symbolica
x = sp.Symbol("x")
varales = [x]
# Definimos el conjunto de operadores/aridades a utilizar
operadores = {"+":2, "-":2, "*":2, "exp":1, "sin":1, "cos":1}
# Definimos el rango para las contsantes numéricas
constantes = [-5.0, 5.5]

# Realizamos una ejecución del algoritmo genético
solucion, _ = evosym(
	datos,
	50,                           # Tamaño de la población
	100,                          # Número de generaciones
	variables,
	operadores,
	constants_range=constantes
	data_prime=datos_deriv,       # Datos de la derivada
	alpha=0.8,                    # Damos un 20% de peso a la derivada
	mutation_method="nodal",      # Elegimos la mutación "nodal"   
	mutation_rate=0.05,           # Probabilidad de mutación del 5%
	fitness_method="mse",         # Función de fitness
	selection_method="torunament",# Selección por torneo binario
	size_penalty="log"            # Penalización de tamaño logarítmica
)

# Mostramos la expresión encontrada
print(expr_from_tree(solucion.tree)) # e.j. cos(x*x) * exp(-x) 
```

En esta ocasión hemos mostrado algunos parámetros adicionales con los que se puede ajustar la ejecución del algoritmo genético. Existen más argumentos posibles y variaciones de los mostrados. Puede encontrarse más información en la documentación de las funciones principales del proyecto.

Finalmente, mostramos cómo obtener funciones reutilizables (objetos Callable de Python) a partir de las soluciones del algoritmo:

```Python
from tfg_evosymbolic.numerify import numerify_tree

# ...

solucion, _ = evosym(
	datos,
	variables,
	operadores,
	# ...
)

f = numerify_tree(solucion.tree)

t, y = datos
y_predict = f(t)

from matplotlib import pyplot as plt
plt.plot(t, y)
plt.plot(t, y_predict)
plt.show()

# Se mostrarían las gráficas de los datos y de la evaluación
# de la expresión encontrada por el algoritmo

```
