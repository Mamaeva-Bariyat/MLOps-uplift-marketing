FROM ubuntu:latest
LABEL authors="bariatmamaeva"

ENTRYPOINT ["top", "-b"]

# Официальный образ Python
FROM python:3.10-slim

# Рабочая директория
WORKDIR /app

# Копируем зависимости
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь проект
COPY . .

# Открываем порт
EXPOSE 8000

# Запуск API
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]