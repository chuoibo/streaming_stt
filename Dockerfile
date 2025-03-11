FROM python:3.10-slim


RUN apt-get update && apt-get install -y \
    ffmpeg \
    gcc

WORKDIR /app

RUN mkdir -p /app/backend /app/frontend

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/asr_socket.py /app/backend/
COPY frontend/index.html /app/frontend/

EXPOSE 5000
EXPOSE 8080

RUN echo '#!/bin/bash\n\
python -m http.server 8080 --directory /app/frontend &\n\
python /app/backend/asr_socket.py\n' > /app/start.sh && \
    chmod +x /app/start.sh

CMD ["/app/start.sh"]