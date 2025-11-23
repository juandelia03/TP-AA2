import pandas as pd
import json

# Cargar el JSON de categorías
with open('token2categoria.json', 'r') as f:
    token2categoria = json.load(f)

# Cargar el CSV
df = pd.read_csv('ejemplo.csv')
df.columns = df.columns.str.strip()  # Limpiar espacios en nombres de columnas

# Lista para almacenar categorías
categorias = []
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
            # Asignar a cada token de la palabra
            for _ in palabra_actual:
                categorias.append(categoria)
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
    for _ in palabra_actual:
        categorias.append(categoria)

# Agregar columna al DataFrame
df['categoria'] = categorias

# Guardar el nuevo CSV
df.to_csv('ejemplo_con_categoria.csv', index=False)
print("CSV con categoría agregado guardado como 'ejemplo_con_categoria.csv'")