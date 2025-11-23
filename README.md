# Proyecto de Procesamiento de Dataset con BERT

Este proyecto procesa un CSV de tokens tokenizados con BERT, agrega categorías usando un JSON, y genera un dataset final con embeddings y características para machine learning.

## Requisitos

- Python 3.8+
- Entorno virtual (venv) recomendado.

Instala dependencias:

```
pip install transformers torch pandas
```

## Archivos Requeridos

- `ejemplo.csv`: CSV original con columnas `instancia_id`, `token_id`, `token`, `punt_inicial`, `punt_final`, `capitalización`. Los tokens deben estar tokenizados con BERT (subwords con `##`).
- `token2categoria.json`: JSON con un diccionario de palabras a categorías

## Archivos Generados

- `ejemplo_con_categoria.csv`: CSV con la columna `categoria` agregada.
- `dataset_final.csv`: Dataset final con atributos (embeddings prev/next, distancias coseno, posiciones) y etiquetas (`categoria`, `punt_inicial`, `punt_final`, `capitalización`).

## Cómo Ejecutar

Sigue este orden:

1. **Agregar categorías**:

   ```
   python add_categoria.py
   ```

   - Lee `ejemplo.csv` y `token2categoria.json`.
   - Genera `ejemplo_con_categoria.csv` con la columna `categoria` (UNE si no encuentra la palabra).

2. **Generar dataset final**:
   ```
   python rf_dataset.py
   ```
   - Lee `ejemplo_con_categoria.csv`.
   - Genera `dataset_final.csv` con embeddings estáticos, distancias coseno, posiciones relativas y etiquetas.
