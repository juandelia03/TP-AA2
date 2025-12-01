# Proyecto de Procesamiento de Dataset con BERT

Este proyecto procesa un CSV de tokens tokenizados con BERT, agrega categorías usando un JSON, y genera un dataset final con embeddings y características para machine learning.

## Requisitos

- Python 3.8+
- Entorno virtual (venv) recomendado.

Instala dependencias:

```
pip install transformers torch pandas
```

## Estructura de Archivos

- `in.csv`: CSV original con columnas `instancia_id`, `token_id`, `token`, `punt_inicial`, `punt_final`, `capitalización`. Los tokens deben estar tokenizados con BERT (subwords con `##`).
- `categorias/`: Carpeta con archivos de categorías:
  - `token2categoria.json`: JSON con un diccionario de palabras a categorías.
  - `nombres_unicos.csv`: Lista de nombres.
  - `apellidos_unicos.csv`: Lista de apellidos.
  - `paises_es.csv`: Lista de países.
  - `siglas_completo.csv`: Lista de siglas.
  - `brand_list_578_rows.csv`: Lista de marcas.

## Archivos Generados

- `categorized.csv`: CSV con la columna `categoria` agregada, más flags `es_nombre`, `es_apellido`, `es_pais`, `es_sigla`, `es_marca` (1 si la palabra está en la lista correspondiente, 0 sino).
- `out.csv`: Dataset final con atributos (embeddings prev/next, distancias coseno, posiciones) y etiquetas (`categoria`, `es_nombre`, `es_apellido`, `es_pais`, `es_sigla`, `es_marca`, `punt_inicial`, `punt_final`, `capitalización`).

## Cómo Ejecutar

Sigue este orden:

1. **Agregar categorías**:

   ```
   python add_categoria.py
   ```

   - Lee `in.csv` y archivos en `categorias/`.
   - Genera `categorized.csv` con la columna `categoria` (UNK si no encuentra la palabra) y flags basados en presencia en listas CSV.

2. **Generar dataset final**:
   ```
   python rf_dataset.py
   ```
   - Lee `categorized.csv`.
   - Genera `out.csv` con embeddings estáticos, distancias coseno, posiciones relativas y etiquetas.
