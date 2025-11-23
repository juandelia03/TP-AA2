from transformers import BertTokenizer, BertModel
import torch
import torch.nn.functional as F
import pandas as pd

model_name = "bert-base-multilingual-cased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
model.eval()

def get_multilingual_token_embedding(token: str):
    """
    Devuelve el embedding (estático) para el token.
    """
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None or token_id == tokenizer.unk_token_id:
        # print(f"❌ El token '{token}' no pertenece al vocabulario de multilingual BERT.")
        return None
    embedding_vector = model.embeddings.word_embeddings.weight[token_id]
    # print(f"✅ Token: '{token}' | ID: {token_id}")
    # print(f"Embedding shape: {embedding_vector.shape}")
    return embedding_vector

def cosine_distance(emb1, emb2):
    return 1 - F.cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()

def create_dataset_csv(input_csv_path: str, output_csv_path: str):
    
    emb_dot = get_multilingual_token_embedding(".")
    
    df = pd.read_csv(input_csv_path)
    all_rows = []
    rows = []
    indice_actual = 0
    emb_promedio = None  # Inicializar como None para evitar error de tipo
    for old in df.itertuples():
        i = old.Index
        row  = {
            'prev_embedding':None,
            'next_embedding':None,
            'embedding_actual':None, # DROPPEAR ESTA KEY LA NECESITO PARA CALCULAR LA DIST PROMEDIO
            'distancia_coseno_prev':None,
            'distancia_coseno_next':None,
            'distancia_promedio':None,
            'posicion_respecto_inicio':None,
            'posicion_respecto_final':None,
            'categoria': df.loc[i, 'categoria'],
            'es_nombre': df.loc[i, 'es_nombre'],
            'es_apellido': df.loc[i, 'es_apellido'],
            'es_pais': df.loc[i, 'es_pais'],
            'es_sigla': df.loc[i, 'es_sigla'],
            'es_marca': df.loc[i, 'es_marca'],
            'punt_inicial': old.punt_inicial,
            'punt_final':old.punt_final,
            'capitalización':old.capitalización
            }
        emb_actual = get_multilingual_token_embedding(df.loc[i, ' token'])
        row['embedding_actual'] = emb_actual
        # Primero asiganar embedding del token anterior
        if indice_actual == 0:
            emb_prev = emb_dot
        else:
            token_prev = df.loc[i - 1, ' token']
            emb_prev = get_multilingual_token_embedding(token_prev)
        dist_prev = cosine_distance(emb_prev, emb_actual)

        ## asignar embedding del token siguiente
        #chequear que no sea el ultimo para no hacer un index error
        if i == len(df) - 1:
            emb_next = emb_dot
        elif df.loc[i+1, 'instancia_id'] != df.loc[i, 'instancia_id']:
            emb_next = emb_dot # se puede sacar promedio entre el punto y el espacio
        else:
            emb_next = get_multilingual_token_embedding(df.loc[i + 1, ' token'])
        dist_next = cosine_distance(emb_next, emb_actual)
        
        # Acumular para promedio
        if emb_promedio is None:
            emb_promedio = emb_actual.clone()
        else:
            emb_promedio += emb_actual

        #asignar la posicion relativa al inicio de la instancia
        row['prev_embedding'] = emb_prev
        row['next_embedding'] = emb_next
        row['distancia_coseno_prev'] = dist_prev
        row['distancia_coseno_next'] = dist_next
        row['posicion_respecto_inicio'] = indice_actual
        rows.append(row)
        indice_actual += 1  
        ## cuando llegamos al ultimo token de la instancia actualizar posiciones_relativas al final y distancia_promedio
        if i == len(df) - 1 or df.loc[i+1, 'instancia_id'] != df.loc[i, 'instancia_id']:
            emb_promedio = emb_promedio / len(rows)
            for j in range(len(rows)):
                rows[j]['distancia_promedio'] = cosine_distance(rows[j]['embedding_actual'], emb_promedio)
                rows[j]['posicion_respecto_final'] = (len(rows) - 1) - j
                all_rows.append(rows[j])
            rows = []
            indice_actual = 0

    ## Ahora creo el dataframe final y lo guardo en un csv
# ...existing code...

    ## Ahora creo el dataframe final y lo guardo en un csv
    #gepeto dice que los embedding hay q pasarlos a lista
    for row in all_rows:
        if row['prev_embedding'] is not None:
            row['prev_embedding'] = row['prev_embedding'].tolist()
        if row['next_embedding'] is not None:
            row['next_embedding'] = row['next_embedding'].tolist()
        if row['embedding_actual'] is not None:
            row['embedding_actual'] = row['embedding_actual'].tolist()
    
    # Crear DataFrame y guardar
    df_output = pd.DataFrame(all_rows)
    df_output.to_csv(output_csv_path, index=False)

create_dataset_csv("categorized.csv", "out.csv")