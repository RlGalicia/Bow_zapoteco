# Bag-of-Words para Verbos Zapotecos

Proyecto que aplica un modelo **Bag-of-Words** (Bolsa de Palabras) para analizar la frecuencia de raíces verbales y pronombres en un corpus de verbos en **zapoteco diidxazá**.

## Objetivo

Cuantificar la presencia de formas verbales válidas (extraídas del archivo **Palabras diidxaza - OrdenadasLower.csv**) dentro de un archivo de conjugaciones (extraídas del archivo **Verbos Zapoteco.csv**), contando cada palabra hasta el primer apóstrofe para agrupar palabras bajo una misma raíz. Esto permite visualizar qué verbos y pronombres son más frecuentes en el corpus.

## Archivos utilizados

| Archivo | Descripción | Rol en el proyecto |
|---------|-------------|--------------------|
| `Verbos Zapoteco.csv` | Conjugaciones completas de verbos en zapoteco (formas correctas e incorrectas, junto con traducciones al español). Cada verbo ocupa un bloque de 5 filas (raíz, español, pronombres españoles, conjugación válida, conjugación no válida). | **Corpus**: texto completo donde se buscan las palabras de interés. |
| `Palabras diidxaza - OrdenadasLower.csv` | Lista de palabras en zapoteco (una por línea) que se consideran **válidas**. Incluye pronombres, raíces verbales y formas conjugadas. | **Vocabulario controlado**: solo estas palabras serán contabilizadas. |

## Metodología

1. **Extracción del vocabulario**: Se toma la única columna de `Palabras diidxaza - OrdenadasLower.csv` y se secciona cada token en el primer apóstrofe (`'`). Por ejemplo, `guuta'lu` → `guuta'`. Esto agrupa todas las personas de un verbo bajo una misma raíz con apóstrofe.
2. **Tokenización del corpus**: Se define un tokenizador personalizado que aplica el mismo seccionado al texto del corpus (`Verbos Zapoteco.csv`), ignorando las filas en español y la conjugación no válida.
3. **Vectorización Bag-of-Words**: Usando `CountVectorizer` de Scikit-learn con el vocabulario controlado, se obtiene una matriz de frecuencias (1 documento × N palabras).
4. **Visualización**: Se genera un gráfico de barras verticales con las frecuencias, coloreadas por rangos y con línea de promedio.

## Resultados

Los resultados obtenidos se encuetran en la carpeta "Resultados"; una imágen la gráfica generada, así como un archivo txt (terminal.txt) que contiene el resultado de lo obtenido en la terminal desde la ejecución del script

## Instalación

### Entorno virtual (venv)

Crear y activar el entorno virtual:

```bash
python -m venv venv
# En Windows:
venv\Scripts\activate
# En Linux/Mac:
source venv/bin/activate
**Instalar dependencias:**
pip install -r requirements.txt
**Ejecución**
python prueba_bow.py
