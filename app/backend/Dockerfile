# Usamos una imagen de Python liviana
FROM python:3.9-slim

# Configuramos la carpeta de trabajo
WORKDIR /code

# Copiamos los requerimientos e instalamos
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Copiamos el resto del código (app.py y el .pt)
COPY . .

# Creamos un usuario no-root (requerido por Hugging Face por seguridad)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Hugging Face espera que escuches en el puerto 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]