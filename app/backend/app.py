from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNFlexible(nn.Module):
    def __init__(self, rnn_sizes, fc_sizes=[], bidirectional=False, rnn_type='GRU'):
        """
        Args:
            rnn_sizes (list): Lista con los hidden_size de cada capa RNN.
                              Ej: [512, 256] crea una capa de 512 seguida de una de 256.
            fc_sizes (list): Lista con los hidden_size de las capas densas intermedias.
                             Ej: [128, 64]. Si está vacío, conecta directo a la salida.
            bidirectional (bool): Si las capas RNN son bidireccionales.
            rnn_type (str): 'GRU' o 'LSTM'.
        """
        super(RNNFlexible, self).__init__()

        self.rnn_type = rnn_type
        self.rnn_layers = nn.ModuleList()

        # 1. Construcción de capas RNN con tamaños variables
        input_dim = 768 # BERT embedding size

        for hidden_dim in rnn_sizes:
            if rnn_type == 'LSTM':
                layer = nn.LSTM(input_dim, hidden_dim, num_layers=1,
                                batch_first=True, bidirectional=bidirectional)
            else:
                layer = nn.GRU(input_dim, hidden_dim, num_layers=1,
                               batch_first=True, bidirectional=bidirectional)

            self.rnn_layers.append(layer)

            # La entrada de la siguiente capa es la salida de la actual
            # Si es bidireccional, el tamaño se duplica
            input_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # 2. Construcción del cabezal Denso (Fully Connected más complejo)
        fc_modules = []

        # Capas densas intermedias (si las hay)
        for hidden_dim in fc_sizes:
            fc_modules.append(nn.Linear(input_dim, hidden_dim))
            fc_modules.append(nn.ReLU())
            input_dim = hidden_dim # Actualizamos para la siguiente capa

        # Capa de proyección final (siempre termina en 10 clases)
        fc_modules.append(nn.Linear(input_dim, 10))

        self.fc = nn.Sequential(*fc_modules)

    def forward(self, x, lengths):
        # x: (batch, max_seq_len, 768)

        # Empaquetamos la secuencia una sola vez al principio
        x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Pasamos el PackedSequence a través de la lista de capas RNN
        # PyTorch permite pasar PackedSequence de una capa RNN a otra directamente
        for layer in self.rnn_layers:
            # x_packed se actualiza en cada paso con la salida de la capa anterior
            x_packed, _ = layer(x_packed)

        # Desempaquetamos solo al final de todas las capas RNN
        out_padded, _ = pad_packed_sequence(x_packed, batch_first=True)

        # out_padded shape: (Batch, Max_Len, Last_Hidden_Size * Directions)

        # Pasamos por el cabezal denso (FC)
        logits = self.fc(out_padded)

        return logits
    
app = Flask(__name__)

CORS(app)

PATH_PESOS = "Bi_3_Layers-512-128-64_LSTM_LR0.001_FCDirecta_epoch_6.pt"
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE = torch.device('cpu')

# Mapeos
MAP_INI = {0: "", 1: "¿"}
MAP_FIN = {0: "", 1: "?", 2: ",", 3: "."}
MAP_CAP = {0: "lower", 1: "title", 2: "upper", 3: "upper"}

#Bert
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')
bert_model.eval()
bert_model.to(DEVICE)

# me traigo la red
rnn_model = RNNFlexible(rnn_sizes=[512, 128, 64], bidirectional=True, rnn_type='LSTM')

# Cargar pesos
checkpoint = torch.load(PATH_PESOS, map_location=DEVICE)
if 'model_state_dict' in checkpoint:
    rnn_model.load_state_dict(checkpoint['model_state_dict'])
else:
    rnn_model.load_state_dict(checkpoint)

rnn_model.to(DEVICE)
rnn_model.eval()

def get_multilingual_token_embedding(token: str):
    token_id = tokenizer.convert_tokens_to_ids(token)
    
    # Verificación de vocabulario
    if token_id is None or token_id == tokenizer.unk_token_id:
        token_id = tokenizer.unk_token_id 
    embedding_vector = bert_model.embeddings.word_embeddings.weight[token_id].to(DEVICE)
    
    return embedding_vector


def predecir(oracion):
    # 1. Tokenizar
    tokens = tokenizer.tokenize(oracion)
    if not tokens: return ""
    
    # 2. Obtener Embeddings UNO POR UNO (Usando tu función)
    lista_embeddings = []
    for t in tokens:
        emb = get_multilingual_token_embedding(t)
        lista_embeddings.append(emb)
    
    # 3. Stackear para crear el tensor (Sequence_Length, 768)
    input_seq = torch.stack(lista_embeddings)
    
    # 4. Agregar dimensión de Batch -> (1, Sequence_Length, 768)
    input_batch = input_seq.unsqueeze(0).to(DEVICE)
    lengths = torch.tensor([len(tokens)], dtype=torch.long)
    
    with torch.no_grad():
        # 5. Pasar por la RNN
        logits = rnn_model(input_batch, lengths) # (1, Seq, 10)
        
        # 6. Obtener predicciones
        p_ini = torch.argmax(logits[:, :, 0:2], dim=2).squeeze().tolist()
        p_fin = torch.argmax(logits[:, :, 2:6], dim=2).squeeze().tolist()
        p_cap = torch.argmax(logits[:, :, 6:10], dim=2).squeeze().tolist()

    # Si es 1 sola palabra, convertir a lista para poder iterar
    if isinstance(p_ini, int): p_ini, p_fin, p_cap = [p_ini], [p_fin], [p_cap]
    
    # 7. Reconstrucción
    palabras_out = []
    
    for i, token in enumerate(tokens):
        word = token
        
        # Recuperar etiquetas
        s_ini = MAP_INI.get(p_ini[i], "")
        s_fin = MAP_FIN.get(p_fin[i], "")
        modo = MAP_CAP.get(p_cap[i], "lower")
        
        # Aplicar mayúsculas
        if modo == "title": word = word.capitalize()
        elif modo == "upper": word = word.upper()
        
        # Unir sub-tokens (##)
        if word.startswith("##"):
            clean_word = word[2:]
            if palabras_out:
                prev = palabras_out.pop() 
                # Si la palabra anterior tenía puntuación pegada, se la sacamos
                # para ponerla al final de este nuevo pedazo (si corresponde)
                if prev[-1] in "?.,": prev = prev[:-1]
                palabras_out.append(prev + clean_word + s_fin)
            else:
                # Caso raro: empieza con ## (ej: error de tokenización), lo tratamos como palabra normal
                palabras_out.append(clean_word + s_fin)
        else:
            # Palabra nueva
            prefix = " " if palabras_out else ""
            palabras_out.append(f"{prefix}{s_ini}{word}{s_fin}")
            
    return "".join(palabras_out)

# Endpoins
@app.route('/predecir', methods=['POST'])
def handle_predecir():
    data = request.get_json()
    
    if not data or 'texto' not in data:
        return jsonify({"error": "Faltó enviar 'texto'"}), 400
    
    try:
        resultado = predecir(data['texto'])
        return jsonify({
            "texto_original": data['texto'],
            "texto_predicho": resultado
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return "El servidor está corriendo. Usá POST /predecir"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
