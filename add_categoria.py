import pandas as pd
import json

# Cargar el JSON de categorías
with open('categorias/token2categoria.json', 'r') as f:
    token2categoria = json.load(f)

# Cargar los sets de flags
nombres_df = pd.read_csv('categorias/nombres_unicos.csv')
nombres_set = set(nombres_df['nombre'].str.lower().tolist())

apellidos_df = pd.read_csv('categorias/apellidos_unicos.csv')
apellidos_set = set(apellidos_df['apellido'].str.lower().tolist())

paises_df = pd.read_csv('categorias/paises_es.csv')
paises_set = set(paises_df['pais_es'].str.lower().tolist())

siglas_df = pd.read_csv('categorias/siglas_completo.csv')
siglas_set = set(siglas_df['sigla'].str.lower().tolist())

marcas_df = pd.read_csv('categorias/brand_list_578_rows.csv')
marcas_set = set(marcas_df['brand'].str.lower().tolist())

# Cargar el CSV
df = pd.read_csv('in.csv')
df.columns = df.columns.str.strip()  # Limpiar espacios en nombres de columnas

# Lista para almacenar categorías y flags
categorias = []
es_nombres = []
es_apellidos = []
es_paises = []
es_siglas = []
es_marcas = []

palabra_actual = []

# Iterar por filas
for idx, row in df.iterrows():
    token = row['token']
    
    if not token.startswith('##'):
        # Procesar palabra anterior si existe
        if palabra_actual:
            # Construir palabra completa
            palabra_completa = ''.join([t[2:] if t.startswith('##') else t for t in palabra_actual])
            # Asignar categoría
            categoria = token2categoria.get(palabra_completa, 'UNK')
            # Asignar flags basadas en presencia en CSV
            es_nombre = 1 if palabra_completa.lower() in nombres_set else 0
            es_apellido = 1 if palabra_completa.lower() in apellidos_set else 0
            es_pais = 1 if palabra_completa.lower() in paises_set else 0
            es_sigla = 1 if palabra_completa.lower() in siglas_set else 0
            es_marca = 1 if palabra_completa.lower() in marcas_set else 0
            # Asignar a cada token de la palabra
            for _ in palabra_actual:
                categorias.append(categoria)
                es_nombres.append(es_nombre)
                es_apellidos.append(es_apellido)
                es_paises.append(es_pais)
                es_siglas.append(es_sigla)
                es_marcas.append(es_marca)
            palabra_actual = []
        
        # Iniciar nueva palabra
        palabra_actual.append(token)
    else:
        # Continuar palabra
        palabra_actual.append(token)

# Procesar la última palabra
if palabra_actual:
    palabra_completa = ''.join([t[2:] if t.startswith('##') else t for t in palabra_actual])
    categoria = token2categoria.get(palabra_completa, 'UNK')
    es_nombre = 1 if palabra_completa.lower() in nombres_set else 0
    es_apellido = 1 if palabra_completa.lower() in apellidos_set else 0
    es_pais = 1 if palabra_completa.lower() in paises_set else 0
    es_sigla = 1 if palabra_completa.lower() in siglas_set else 0
    es_marca = 1 if palabra_completa.lower() in marcas_set else 0
    for _ in palabra_actual:
        categorias.append(categoria)
        es_nombres.append(es_nombre)
        es_apellidos.append(es_apellido)
        es_paises.append(es_pais)
        es_siglas.append(es_sigla)
        es_marcas.append(es_marca)

# Agregar columnas al DataFrame
df['categoria'] = categorias
df['es_nombre'] = es_nombres
df['es_apellido'] = es_apellidos
df['es_pais'] = es_paises
df['es_sigla'] = es_siglas
df['es_marca'] = es_marcas

# Guardar el nuevo CSV
df.to_csv('categorized.csv', index=False)
print("CSV con categoría y flags guardado como 'categorized.csv'")