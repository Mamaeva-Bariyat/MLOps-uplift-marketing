# Uplift Marketing MLOps Project

Проект для uplift-моделирования на датасете Hillstrom (Customer Retention).  
Цель — предсказывать прирост конверсии от маркетинговых предложений (BOGO / Discount), чтобы таргетировать только тех клиентов, у которых предложение реально повышает вероятность покупки.


## Как запустить проект

### 1. Клонирование и установка

```bash
git clone https://github.com/ВАШ_НИК/uplift-marketing-mlops.git
cd uplift-marketing-mlops 
```

### 2. Виртуальное окружение

```bash
python -m venv .venv
source .venv/bin/activate          # Mac/Linux
# или .venv\Scripts\activate.bat   # Windows
```

### 3. Установка зависимостей
```bash
pip install -r requirements.txt
pip install -r backend/requirements.txt     # для API
pip install -r frontend/requirements.txt    # для дашборда
```

### 5. Запуск frontend (Streamlit)
```bash
streamlit run frontend/main.py
````
Frontend Dashboard: http://localhost:8501


### 6. Запуск backend (FastAPI)
```bash
uvicorn backend.main:app --reload --port 8000
# или на другом порту, если 8000 занят:
# uvicorn backend.main:app --reload --port 8001
```
Документация API: http://127.0.0.1:8000/docs

### 7. Запуск всего проекта в Docker (опционально)
```bash
docker-compose up --build
```