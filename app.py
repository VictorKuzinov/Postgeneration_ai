# -*- coding: utf-8 -*-
"""
Генератор постов для товара: Flask + нейросеть + ВКонтакте.

Как запустить:
    1. pip install -r requirements.txt
    2. В файле .env заполните ключи (см. VK_SETUP.md — как их получить):
         API_KEY=sk-or-...        — ключ OpenRouter (нейросеть)
         VK_TOKEN=...             — токен сообщества ВКонтакте
         VK_GROUP_ID=123456       — числовой ID сообщества
    3. python app.py
    4. Открыть в браузере http://127.0.0.1:5000

Что умеет:
    - генерирует пост по ссылке на товар с помощью нейросети
    - копирует пост, сохраняет в избранное (файл favorites.json)
    - публикует пост в сообществе ВКонтакте сразу или отложенно
      (отложенные посты создаются самим ВК — раздел «Отложенные записи»)

Полезные ссылки:
    Модели OpenRouter:  https://openrouter.ai/models
    API ВКонтакте:      https://dev.vk.com/ru/reference
    Подключение ВК:     VK_SETUP.md (инструкция рядом с этим файлом)
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, render_template, request

# Читает переменные из файла .env (API_KEY, MODEL, VK_TOKEN, VK_GROUP_ID)
load_dotenv()

app = Flask(__name__)

# =====================================================================
# НАСТРОЙКИ НЕЙРОСЕТИ (OpenRouter)
# =====================================================================
# OpenRouter — единый доступ ко множеству моделей в формате OpenAI API.

API_URL = "https://openrouter.ai/api/v1/chat/completions"

# API-ключ хранится в файле .env и в код не попадает
API_KEY = os.getenv("API_KEY", "")

# Какую модель использовать (формат: вендор/модель).
# По умолчанию — бесплатная модель (суффикс :free).
# Другую модель можно задать в .env строкой:  MODEL=openai/gpt-4o-mini
# Платные модели: https://openrouter.ai/models
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
# НАСТРОЙКИ ВКОНТАКТЕ
# =====================================================================
# Токен и ID сообщества — тоже из файла .env.
# Подробная инструкция по получению — в файле VK_SETUP.md.

VK_API_URL = "https://api.vk.com/method/"
VK_TOKEN = os.getenv("VK_TOKEN", "")
VK_GROUP_ID = os.getenv("VK_GROUP_ID", "").strip().lstrip("-")
VK_API_VERSION = "5.199"

# Файл, где хранятся посты «В избранном» (просто JSON, база не нужна)
FAVORITES_FILE = Path("favorites.json")


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
# ИЗБРАННОЕ (посты сохраняются в файл favorites.json)
# =====================================================================
def load_favorites():
    """Читает список избранных постов из файла."""
    try:
        with open(FAVORITES_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            return data if isinstance(data, list) else []
    except (FileNotFoundError, json.JSONDecodeError):
        return []  # файла нет или он битый — считаем, что избранное пустое


def save_favorites(favorites):
    """Сохраняет список избранных постов в файл."""
    with open(FAVORITES_FILE, "w", encoding="utf-8") as file:
        json.dump(favorites, file, ensure_ascii=False, indent=2)


def add_to_favorites(text, link):
    """Добавляет пост в избранное. Возвращает (список, сообщение)."""
    favorites = load_favorites()
    favorites.insert(0, {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),  # уникальный номер
        "text": text,
        "link": link,
        "created": datetime.now().strftime("%d.%m.%Y %H:%M"),
    })
    save_favorites(favorites)
    return favorites, "Пост сохранён в избранное"


def delete_from_favorites(fav_id):
    """Удаляет пост из избранного по его номеру."""
    favorites = [f for f in load_favorites() if f.get("id") != fav_id]
    save_favorites(favorites)
    return favorites, "Пост удалён из избранного"


# =====================================================================
# ПУБЛИКАЦИЯ В ВКОНТАКТЕ
# =====================================================================
def publish_to_vk(text, publish_at=None):
    """
    Публикует пост в сообществе ВКонтакте.

    Параметры:
        text       — текст поста
        publish_at — объект datetime для отложенной записи
                     (None — опубликовать сразу)

    ВКонтакте умеет сама хранить отложенные записи: если передать
    publish_date, пост появится в разделе «Отложенные записи»
    сообщества и будет опубликован автоматически в нужное время.
    Наш сервер для этого работать не обязан — можно даже выключить.
    """
    if not VK_TOKEN or not VK_GROUP_ID:
        return ("ВКонтакте не настроено. Заполните VK_TOKEN и VK_GROUP_ID "
                "в файле .env — инструкция в VK_SETUP.md.")

    params = {
        # owner_id сообщества всегда со знаком минус
        "owner_id": f"-{VK_GROUP_ID}",
        "message": text,
        "access_token": VK_TOKEN,
        "v": VK_API_VERSION,
    }

    # Отложенная публикация: время в формате Unix (секунды)
    if publish_at is not None:
        params["publish_date"] = int(publish_at.timestamp())

    try:
        response = requests.get(
            VK_API_URL + "wall.post",
            params=params,
            timeout=30,
        )
    except requests.exceptions.RequestException:
        return "Не удалось связаться с ВКонтакте. Проверьте интернет."

    data = response.json()

    # ВК всегда отвечает {"error": {...}} при проблемах
    if "error" in data:
        return f"Ошибка ВКонтакте: {data['error'].get('error_msg', 'неизвестная ошибка')}"

    post_id = data.get("response", {}).get("post_id")

    if publish_at is not None:
        return (f"Пост отложен! ВКонтакте опубликует его "
                f"{publish_at.strftime('%d.%m.%Y в %H:%M')} "
                f"(запись №{post_id} в разделе «Отложенные записи»).")
    return f"Пост опубликован в сообществе (запись №{post_id})."


def split_result(text):
    """
    Делит результат на (сообщение_об_успехе, ошибка).
    Если текст похож на ошибку — он уйдёт в красный блок.
    """
    if text.startswith(("Ошибка", "ВКонтакте не настроено", "Не удалось")):
        return "", text
    return text, ""


# =====================================================================
# СТРАНИЦА САЙТА
# =====================================================================
@app.route("/", methods=["GET", "POST"])
def index():
    """
    Главная страница: генерация поста, избранное, публикация в ВК.
    Вся обработка — здесь, через поле action в форме.
    """
    post_text = ""       # текст поста, который сейчас показан на странице
    link = ""            # ссылка на товар для этого поста
    message = ""         # зелёное сообщение об успехе
    error = ""           # красное сообщение об ошибке

    if request.method == "POST":
        action = request.form.get("action", "generate")

        # ---------- Сгенерировать новый пост ----------
        if action == "generate":
            link = request.form.get("link", "").strip()
            mood_key = request.form.get("mood", "friendly")

            if not link:
                error = "Пожалуйста, вставьте ссылку на товар."
            elif not link.startswith(("http://", "https://")):
                error = "Ссылка должна начинаться с http:// или https://"
            else:
                settings = load_voice()
                prompt = build_prompt(link, mood_key, settings)
                post_text, error = ask_neural_network(prompt)

        # ---------- Действия с уже готовым постом ----------
        else:
            # Текст поста приезжает из скрытого поля формы
            post_text = request.form.get("post_text", "")
            link = request.form.get("link", "")

            if action == "favorite" and post_text:
                _, message = add_to_favorites(post_text, link)

            elif action == "publish" and post_text:
                message, error = split_result(publish_to_vk(post_text))

            elif action == "schedule" and post_text:
                # Время из поля datetime-local (например 2026-10-21T15:30)
                publish_at_raw = request.form.get("publish_at", "")
                try:
                    publish_at = datetime.fromisoformat(publish_at_raw)
                    if publish_at <= datetime.now():
                        error = "Выберите время в будущем."
                    else:
                        message, error = split_result(
                            publish_to_vk(post_text, publish_at)
                        )
                except ValueError:
                    error = "Укажите дату и время для отложенной записи."

            elif action in ("use", "edit") and post_text:
                # "use" — вставить пост из избранного в блок «Готовый пост»
                # "edit" — сохранить текст после редактирования в textarea.
                # В обоих случаях новый текст уже лежит в post_text —
                # просто покажем его на странице.
                pass

            elif action == "delete_fav":
                favorites, message = delete_from_favorites(
                    request.form.get("fav_id", "")
                )

    return render_template(
        "index.html",
        post=post_text,
        link=link,
        error=error,
        message=message,
        favorites=load_favorites(),
        # Список настроений для выпадающего списка на странице
        mood_options=[
            {"key": key, "label": value["label"]}
            for key, value in MOODS.items()
        ],
        # Нужно ли показывать блок «Готовый пост»
        show_actions=bool(post_text),
    )


if __name__ == "__main__":
    # debug=True — перезапуск сервера при изменении кода
    app.run(debug=True)
