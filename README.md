
# Video2Text — конвертер видео в текст

## 🚀 Запуск локально

```bash
git clone <your_repo_url>
cd video2text
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

## 🌐 Деплой на Render

1. Создай новый Web Service.
2. Подключи этот репозиторий.
3. В Build Command: оставить пустым или `pip install -r requirements.txt`.
4. В Start Command: `web: python app.py`.

Приложение будет запущено на порту 10000+ и доступно в браузере.
