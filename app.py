# -*- coding: utf-8 -*-
"""
Генератор постов для товара: Flask + нейросеть через OpenRouter.

Как запустить:
    1. pip install -r requirements.txt
    2. В файле .env должен быть ключ:  API_KEY=sk-or-ваш_ключ
    3. python app.py
    4. Открыть в браузере http://127.0.0.1:5000

Как работает:
    пользователь вставляет ссылку на товар -> сайт отправляет её
    нейросети OpenRouter (модель сама открывает страницу и изучает товар)
    -> модель пишет оригинальный пост -> пост появляется на странице.

Полезные ссылки:
    Каталог моделей и цен: https://openrouter.ai/models
    Документация:          https://openrouter.ai/docs
"""

import os
import re

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request

# Читает переменные из файла .env (API_KEY, MODEL)
load_dotenv()

app = Flask(__name__)

# =====================================================================
# НАСТРОЙКИ НЕЙРОСЕТИ (OpenRouter)
# =====================================================================
# OpenRouter — это единый доступ ко множеству моделей (OpenAI, Anthropic,
# Google и другие) по одному ключу и одному адресу, в формате OpenAI API.

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# API-ключ хранится в файле .env и в код не попадает
API_KEY = os.getenv("API_KEY", "")

# Какую модель использовать (формат: вендор/модель).
# По умолчанию — бесплатная модель (суффикс :free).
# Другую модель можно задать в .env строкой:  MODEL=openai/gpt-4o-mini
# Платные модели: https://openrouter.ai/models
# Бесплатные списывает только время ожидания — лимиты строже.
MODEL = os.getenv("MODEL", "minimax/minimax-m3.1-flash:free")

# Сколько секунд ждать ответ (поиск в интернете может быть долгим)
TIMEOUT = 120

# Разрешает ли модели ходить в интернет за ссылкой.
# True — к имени модели добавляется ":online", и OpenRouter включает
# веб-поиск: модель открывает страницу товара и пишет пост по реальным
# данным. ВАЖНО: поиск в интернете оплачивается отдельно (нужен баланс),
# на бесплатных моделях может не работать.
# False — модель пишет пост по ссылке без открытия страницы.
USE_WEB_SEARCH = False


# =====================================================================
# НАСТРОЙКИ ИЗ ФАЙЛА voice.md
# =====================================================================
# Тон, длина, эмодзи и заголовок поста задаются в voice.md.
# Чтобы изменить стиль постов — правьте voice.md, код трогать не нужно.

VOICE_FILE = "voice.md"

# Значения по умолчанию (если в voice.md чего-то нет)
DEFAULT_VOICE = {
    "Общий тон": "дружелюбный",
    "Длина поста": "до 300 символов",
    "Количество эмодзи": "3",
    "Стиль заголовка": "обычный",
}


def load_voice():
    """Читает voice.md и возвращает словарь с настройками поста."""
    settings = dict(DEFAULT_VOICE)

    try:
        with open(VOICE_FILE, "r", encoding="utf-8") as file:
            for line in file:
                # Ищем строки вида:  - **Ключ:** значение
                match = re.search(r"\*\*(.+?)[:：]\*\*\s*(.+)", line)
                if match:
                    key = match.group(1).strip()
                    value = match.group(2).strip()
                    if key in settings:
                        settings[key] = value
    except FileNotFoundError:
        pass  # файла нет — используем значения по умолчанию

    return settings


# =====================================================================
# НАСТРОЙКИ ПОСТА (редактируйте под себя)
# =====================================================================

# Настроения поста — выбираются в выпадающем списке на странице.
# Каждое настроение задаёт "эмоцию" текста для нейросети.
MOODS = {
    "friendly": {
        "label": "Дружелюбное",
        "prompt": "тёплое и искреннее, как совет хорошего друга",
    },
    "business": {
        "label": "Деловое",
        "prompt": "сдержанное, профессиональное, с акцентом на пользу и качество",
    },
    "excited": {
        "label": "Воодушевляющее",
        "prompt": "энергичное, яркое, восторженное — хочется купить прямо сейчас",
    },
    "urgent": {
        "label": "Срочное (акция)",
        "prompt": "с чувством срочности и выгоды: ограниченное предложение, нужно успеть",
    },
}


def build_prompt(link, mood_key, settings):
    """
    Собирает промпт (задание для нейросети) из ссылки,
    выбранного настроения и настроек из voice.md.
    """
    mood = MOODS.get(mood_key, MOODS["friendly"])

    # Если веб-поиск выключен — не просим модель открывать ссылку,
    # а просим написать пост, опираясь на адрес и свои знания.
    if USE_WEB_SEARCH:
        first_step = """1. Сначала открой ссылку и изучи товар: название, характеристики,
   преимущества, цену — всё, что найдёшь на странице."""
    else:
        first_step = """1. Определи по ссылке, о каком товаре речь (название магазина,
   категория товара в адресе), и напиши правдоподобный живой пост.""" 

    return f"""
Напиши рекламный пост для социальных сетей о товаре по ссылке.

Ссылка на товар: {link}

Как написать пост:
{first_step}
2. Напиши пост на русском языке. Каждый раз пост получается
   оригинальным и интересным — не используй шаблонные фразы
   вроде "В наше время хочется..." или "Каждый мечтает...".
3. Настроение поста: {mood["prompt"]}.
4. Заголовок: стиль "{settings["Стиль заголовка"]}".
5. Длина: примерно {settings["Длина поста"]}.
6. Эмодзи: используй примерно {settings["Количество эмодзи"]} штук,
   расставь их органично, не подряд.
7. Общий тон общения: {settings["Общий тон"]}.
8. Заканчивай пост коротким призывом к действию
   (написать в личку, перейти по ссылке, успеть купить).
9. В конце поста отдельной строкой добавь саму ссылку: {link}

Верни только текст поста — без пояснений, без кавычек вокруг ответа.
""".strip()


# =====================================================================
# ЗАПРОС К НЕЙРОСЕТИ
# =====================================================================
def extract_post_text(response_json):
    """
    Достаёт текст поста из ответа API.
    Ответ в формате OpenAI Chat Completions:
    {"choices": [{"message": {"content": "текст поста"}}]}
    """
    choices = response_json.get("choices", [])
    if choices:
        content = choices[0].get("message", {}).get("content", "")
        # Иногда content приходит списком кусков текста
        if isinstance(content, list):
            return "\n".join(
                block.get("text", "") for block in content
            ).strip()
        return str(content).strip()

    return ""


def ask_neural_network(prompt):
    """
    Отправляет промпт в OpenRouter и возвращает (текст, ошибка).
    Если всё хорошо — текст поста и пустая ошибка.
    Если что-то пошло не так — пустой текст и описание проблемы.
    """
    if not API_KEY:
        return "", "Не найден API-ключ. Пропишите его в файле .env: API_KEY=sk-or-..."

    # Имя модели для запроса: с ":online" — с веб-поиском
    model_name = MODEL
    if USE_WEB_SEARCH:
        model_name = f"{MODEL}:online"

    # Тело запроса — стандартный формат OpenAI Chat Completions
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt},
        ],
    }

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                # Заголовки OpenRouter: сайт и название проекта (необязательные)
                "HTTP-Referer": "http://localhost:5000",
                "X-OpenRouter-Title": "Post Generator",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=TIMEOUT,
        )
    except requests.exceptions.Timeout:
        return "", "Нейросеть не успела ответить. Попробуйте ещё раз."
    except requests.exceptions.RequestException:
        return "", "Не удалось связаться с сервером. Проверьте интернет."

    # Обрабатываем ошибки API понятным для новичка языком
    if response.status_code == 401:
        return "", "API-ключ не подошёл. Проверьте его в файле .env."
    if response.status_code == 402:
        return "", "Закончились средства на балансе OpenRouter. Пополните: https://openrouter.ai/credits"
    if response.status_code == 403:
        return "", "Сервис запретил доступ (код 403). Проверьте интернет, VPN или прокси."
    if response.status_code == 429:
        return "", "Слишком много запросов. Подождите немного и повторите."
    if response.status_code != 200:
        return "", f"Ошибка от нейросети (код {response.status_code})."

    post = extract_post_text(response.json())

    if not post:
        return "", "Нейросеть вернула пустой ответ. Попробуйте ещё раз."

    return post, ""


# =====================================================================
# СТРАНИЦА САЙТА
# =====================================================================
@app.route("/", methods=["GET", "POST"])
def index():
    """Главная страница: форма со ссылкой и блок с готовым постом."""
    post_text = ""
    error = ""
    link = ""

    if request.method == "POST":
        link = request.form.get("link", "").strip()
        mood_key = request.form.get("mood", "friendly")

        # Проверка: ссылка обязательна (в форме помечена звёздочкой)
        if not link:
            error = "Пожалуйста, вставьте ссылку на товар."
        elif not link.startswith(("http://", "https://")):
            error = "Ссылка должна начинаться с http:// или https://"
        else:
            settings = load_voice()
            prompt = build_prompt(link, mood_key, settings)
            post_text, error = ask_neural_network(prompt)

    return render_template(
        "index.html",
        post=post_text,
        error=error,
        link=link,
        # Список настроений для выпадающего списка на странице
        mood_options=[
            {"key": key, "label": value["label"]}
            for key, value in MOODS.items()
        ],
    )


if __name__ == "__main__":
    # debug=True — перезапуск сервера при изменении кода
    app.run(debug=True)
