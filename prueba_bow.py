import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df_documento = pd.read_csv('Verbos Zapoteco.csv') #Documento que actuara como corpus
print(f" {len(df_documento)} Columnas del documento:", df_documento.columns.tolist())
print(df_documento.head(10))

df_verbos = pd.read_csv('Palabras diidxaza - OrdenadasLower.csv') #Documento con las palabrtas válidas
print(f" {len(df_verbos)} Columnas de verbos:", df_verbos.columns.tolist())
print(df_verbos.head(10))

vocabulario = set()

for celda in df_verbos.iloc[:, 0].dropna(): # La única columna de df_verbos
    tokens = str(celda).strip().lower().split() # Casteo a string y separamos por espacios
    for token in tokens:
        # Hasta el primer apóstrofe { ' }
        apostrofe_verbo = token.find("'")
        if apostrofe_verbo != -1:
            # Tomamos la palabra apóstrofe incluido
            token_apostrofe = token[:apostrofe_verbo+1]   # "guuta'lu" → "guuta'"
        else:
            token_apostrofe = token                     # sin apóstrofe, queda igual

        vocabulario.add(token_apostrofe)

print(f"Tamaño del vocabulario: {len(vocabulario)}")
print("Muestra ordenada:", sorted(list(vocabulario))[:15]) #Se muestran 15

def tokenizador_apostrofe(texto): # Tokenizador personalizado
    tokens = texto.strip().lower().split()
    tokens_apostrofe = []
    for token in tokens:
        pos = token.find("'")
        if pos != -1:
            tokens_apostrofe.append(token[:pos+1])  # incluye el apóstrofe
        else:
            tokens_apostrofe.append(token)
    return tokens_apostrofe

texto_corpus = ' '.join(
    str(celda) for col in df_documento.columns
    for celda in df_documento[col].dropna()
)
corpus = [texto_corpus]   # Covertimos el csv una lista con un solo string

# Crear el vectorizador usando el vocabulario y tokenizador apóstrofe
vectorizador = CountVectorizer(
    vocabulary=vocabulario,          # las palabras que nos interesan
    tokenizer=tokenizador_apostrofe,       # aplica la misma lógica al corpus
    lowercase=True                      #minusculas
)

X = vectorizador.fit_transform(corpus)
print("Dimensión de matriz:", X.shape)  # (1, len(vocabulario))

#graficando
frecuencias = X.toarray()[0]
palabras = vectorizador.get_feature_names_out()

indices_top = frecuencias.argsort()[:][::-1]
top_palabras = [palabras[i] for i in indices_top]
top_freq = frecuencias[indices_top]

print("Lista de palabras frecuentes en el documento:")
for palabra, freq in zip(top_palabras, top_freq):
    print(f"  {palabra}: {freq}")


indices_ordenados = np.argsort(frecuencias)[::-1]
palabras_ordenadas = [palabras[i] for i in indices_ordenados]
frecuencias_ordenadas = frecuencias[indices_ordenados]

fig, ax = plt.subplots(figsize=(20, 8))

def color_por_frecuencia(f):
    if f >= 100:      return '#d73027'   
    elif f >= 15:     return '#fc8d59'  
    elif f >= 8:      return '#f7dc6f'   
    elif f >= 3:      return '#85c1e9'   
    else:             return '#bdc3c7'   

colores = [color_por_frecuencia(f) for f in frecuencias_ordenadas]

barras = ax.bar(
    range(len(palabras_ordenadas)),
    frecuencias_ordenadas,
    color=colores,
    edgecolor='gray',
    linewidth=0.5
)

ax.set_xticks(range(len(palabras_ordenadas)))
ax.set_xticklabels(palabras_ordenadas, rotation=90, fontsize=8)
ax.set_xlim(-0.5, len(palabras_ordenadas) - 0.5)

ax.set_xlabel('Vocabulario')
ax.set_ylabel('Frecuencia')
ax.set_title('Frecuencia de verbos Zapoteco en todo el corpus')

promedio = np.mean(frecuencias) # Línea de promedio
ax.axhline(promedio, color='blue', linestyle='--', linewidth=1, alpha=0.7, label=f'Promedio ({promedio:.1f})')
ax.legend()

if len(palabras_ordenadas) <= 30:
    for i, freq in enumerate(frecuencias_ordenadas):
        if freq > 0:
            ax.text(i, freq + 0.3, str(freq), ha='center', fontsize=7)
else:
    for i, freq in enumerate(frecuencias_ordenadas):    # Mostrar etiquetas cada 5 barras
        if freq > 0 and i % 5 == 0:
            ax.text(i, freq + 0.5, str(freq), ha='center', fontsize=7)

plt.tight_layout()
plt.show()