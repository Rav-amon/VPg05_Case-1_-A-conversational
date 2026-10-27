"""Телеграм-бот, который общается с пользователем и помнит его через Pinecone.

Каждое текстовое сообщение сохраняется менеджером ``PineconeManager``.
Перед ответом бот достаёт из того же хранилища факты именно этого пользователя
и передаёт их модели как долговременную память.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence

import telebot
import telebot.apihelper as telegram_api
from dotenv import load_dotenv
from openai import OpenAI

from pinecone_manager import (
    COSINE_SIMILARITY_THRESHOLD,
    MEMORY_ACTION_SKIP,
    MEMORY_ACTION_UPDATE,
    PineconeManager,
    PineconeManagerError,
    StoredDocument,
    decide_memory_action,
    new_memory_id,
    read_env,
    require_config,
    require_cosine_threshold,
)

logger = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN_ENV = 'TELEGRAM_BOT_TOKEN'
OPENAI_API_KEY_ENV = 'OPENAI_API_KEY'
OPENAI_BASE_URL_ENV = 'OPENAI_BASE_URL'
OPENAI_CHAT_MODEL_ENV = 'OPENAI_CHAT_MODEL'
DEFAULT_CHAT_MODEL = 'gpt-4o-mini'
# Поле схемы индекса, которое Pinecone сам превращает в вектор.
MEMORY_TEXT_FIELD = 'chunk_text'

# Сколько прошлых реплик держать в текущем сеансе, чтобы диалог не рвался.
MAX_HISTORY_MESSAGES = 10
# Сколько фрагментов памяти подмешивать в ответ.
MEMORY_TOP_K = 5
# Лимит Telegram чуть ниже 4096, чтобы разметка не упиралась в край.
TELEGRAM_MESSAGE_LIMIT = 4000
# HTTP-таймаут должен быть больше времени ожидания getUpdates.
POLLING_HTTP_TIMEOUT = 30
# Короткое ожидание реже обрывается промежуточным узлом сети.
LONG_POLLING_TIMEOUT = 10
# Через сколько секунд выбрасывать зависшую HTTP-сессию Telegram.
POLLING_SESSION_SECONDS = 120
MAX_REPLY_TOKENS = 800
CHAT_TEMPERATURE = 0.7
USER_ROLE = 'user'
ASSISTANT_ROLE = 'assistant'
MEMORY_ROLE = 'user'

SYSTEM_PROMPT = (
    'Ты внимательный помощник в Telegram. Отвечай на русском языке, '
    'ясно и по делу. Используй сохранённые факты о пользователе, когда они '
    'относятся к вопросу. Не приписывай пользователю сведения, которых нет '
    'ни в памяти, ни в текущем диалоге. Если факта не хватает, спроси.'
)
START_TEXT = (
    'Привет. Я помощник и запоминаю то, что вы о себе рассказываете. '
    'Можно просто писать сообщения. Команда /forget стирает вашу память.'
)
FORGET_TEXT = 'Я удалил всё, что помнил о вас в этом хранилище.'
UNSUPPORTED_TEXT = 'Я сохраняю и обсуждаю только текстовые сообщения.'
EMPTY_TEXT = 'Напишите сообщение текстом.'
UNKNOWN_COMMAND_TEXT = 'Такой команды нет. Доступны /start и /forget.'
CHAT_ERROR_TEXT = 'Сейчас не получилось подготовить ответ. Попробуйте ещё раз.'
MEMORY_ERROR_SUFFIX = '\n\nСообщение не удалось записать в память.'
EMPTY_ANSWER_TEXT = 'Я не смог сформулировать ответ. Напишите ещё раз.'

UNSUPPORTED_CONTENT_TYPES = (
    'audio',
    'document',
    'photo',
    'sticker',
    'video',
    'voice',
    'location',
    'contact',
)


class TelegramBotError(Exception):
    """Ошибка бота с кодом, сообщением и подробностями.

    Attributes:
        code: Машинный код ошибки.
        message: Текст для журнала.
        details: Дополнительные поля без секретов.
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Сохраняет код и подробности ошибки.

        Args:
            code: Машинный код.
            message: Текст исключения.
            details: Дополнительные поля. ``None`` становится пустым словарём.
        """
        self.code = code
        self.message = message
        self.details = dict(details or {})
        super().__init__(message)


def user_memory_filter(user_id: int) -> dict[str, object]:
    """Собирает фильтр Pinecone только по записям одного пользователя.

    Args:
        user_id: Числовой идентификатор Telegram.

    Returns:
        Фильтр метаданных для поиска и удаления.
    """
    return {'user_id': {'$eq': str(user_id)}}


def build_user_metadata(user: telebot.types.User) -> dict[str, object]:
    """Собирает метаданные слота памяти из профиля Telegram.

    Args:
        user: Отправитель сообщения.

    Returns:
        Поля, которые Pinecone может сохранить рядом с текстом.
    """
    metadata: dict[str, object] = {
        'user_id': str(user.id),
        'role': MEMORY_ROLE,
    }
    if isinstance(user.first_name, str) and user.first_name.strip() != '':
        metadata['first_name'] = user.first_name.strip()
    if isinstance(user.username, str) and user.username.strip() != '':
        metadata['username'] = user.username.strip()
    return metadata


def profile_text(user: telebot.types.User) -> str:
    """Формулирует устойчивый факт о том, кто пишет боту.

    Args:
        user: Отправитель команды /start.

    Returns:
        Текст для долговременной памяти.
    """
    name = user.first_name.strip() if isinstance(user.first_name, str) else ''
    if name == '':
        name = 'без имени'
    parts = [f'Имя пользователя в Telegram: {name}.', f'Идентификатор: {user.id}.']
    if isinstance(user.username, str) and user.username.strip() != '':
        parts.append(f'Имя пользователя: @{user.username.strip()}.')
    return ' '.join(parts)


def format_memories(documents: Sequence[StoredDocument]) -> str:
    """Склеивает найденные фрагменты в текст для системной подсказки.

    Args:
        documents: Документы поиска, ближайшие первыми.

    Returns:
        Маркированный список. Пустая строка, если текстов нет.
    """
    lines: list[str] = []
    for document in documents:
        raw_text = document.fields.get(MEMORY_TEXT_FIELD)
        if not isinstance(raw_text, str):
            continue
        cleaned = raw_text.strip()
        if cleaned == '' or cleaned in lines:
            continue
        lines.append(cleaned)
    return '\n'.join(f'- {line}' for line in lines)


def build_chat_messages(
    memory_text: str,
    history: Sequence[Mapping[str, str]],
    user_text: str,
) -> list[dict[str, str]]:
    """Собирает запрос к модели: инструкция, память, недавний диалог и реплика.

    Args:
        memory_text: Уже отформатированные факты. Пустая строка допустима.
        history: Предыдущие реплики сеанса с полями ``role`` и ``content``.
        user_text: Текущее сообщение пользователя.

    Returns:
        Список сообщений для Chat Completions.
    """
    system_text = SYSTEM_PROMPT
    if memory_text != '':
        system_text = (
            f'{SYSTEM_PROMPT}\n\nЧто уже известно об этом пользователе:\n'
            f'{memory_text}'
        )
    messages = [{'role': 'system', 'content': system_text}]
    for item in history:
        role = item.get('role', '')
        content = item.get('content', '')
        if role not in (USER_ROLE, ASSISTANT_ROLE) or content == '':
            continue
        messages.append({'role': role, 'content': content})
    messages.append({'role': USER_ROLE, 'content': user_text})
    return messages


def trim_history(
    turns: Sequence[Mapping[str, str]],
    limit: int,
) -> list[dict[str, str]]:
    """Оставляет только последние реплики сеанса.

    Args:
        turns: Полная история.
        limit: Максимум сообщений.

    Returns:
        Новый список не длиннее ``limit``.
    """
    kept = [
        {'role': item['role'], 'content': item['content']}
        for item in turns
        if item.get('role') in (USER_ROLE, ASSISTANT_ROLE)
        and isinstance(item.get('content'), str)
        and item.get('content') != ''
    ]
    if len(kept) <= limit:
        return kept
    return kept[-limit:]


def split_telegram_text(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Делит длинный ответ на части, которые принимает Telegram.

    Args:
        text: Исходный ответ.
        limit: Максимальная длина одной части.

    Returns:
        Непустой список частей.
    """
    if text.strip() == '':
        return [EMPTY_ANSWER_TEXT]
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind('\n', 0, limit)
        if cut < 1:
            cut = limit
        parts.append(rest[:cut].strip())
        rest = rest[cut:].lstrip()
    if rest.strip() != '':
        parts.append(rest.strip())
    return [part for part in parts if part != '']


class TelegramAssistant:
    """Принимает сообщения Telegram и хранит сведения о пользователе в Pinecone."""

    def __init__(
        self,
        token: str,
        memory: PineconeManager,
        chat_client: OpenAI,
        chat_model: str,
    ) -> None:
        """Связывает бота, память и клиент диалоговой модели.

        Args:
            token: Токен Telegram-бота.
            memory: Менеджер векторного хранилища.
            chat_client: Клиент OpenAI для ответов.
            chat_model: Имя диалоговой модели.
        """
        self._memory = memory
        self._chat_client = chat_client
        self._chat_model = chat_model
        self._bot = telebot.TeleBot(token)
        self._history: dict[int, list[dict[str, str]]] = {}
        self._lock = threading.Lock()
        self._register_handlers()

    @classmethod
    def from_env(cls) -> TelegramAssistant:
        """Создаёт бота из переменных окружения и файла ``.env``.

        Returns:
            Готовый помощник.

        Raises:
            TelegramBotError: Если не задан токен или ключ OpenAI.
            PineconeManagerError: Если не открывается индекс Pinecone.
        """
        load_dotenv()
        token = require_config(read_env(TELEGRAM_BOT_TOKEN_ENV), TELEGRAM_BOT_TOKEN_ENV)
        api_key = require_config(read_env(OPENAI_API_KEY_ENV), OPENAI_API_KEY_ENV)
        base_url = read_env(OPENAI_BASE_URL_ENV)
        chat_model = read_env(OPENAI_CHAT_MODEL_ENV) or DEFAULT_CHAT_MODEL
        if base_url is None:
            chat_client = OpenAI(api_key=api_key)
        else:
            chat_client = OpenAI(api_key=api_key, base_url=base_url)
        return cls(
            token=token,
            memory=PineconeManager(),
            chat_client=chat_client,
            chat_model=chat_model,
        )

    def run(self) -> None:
        """Запускает приём сообщений, пока процесс не остановлен.

        Telegram иногда закрывает длинный getUpdates без ответа.
        Короткий опрос и повтор запроса поднимают приём сообщений снова.
        """
        logger.info('Телеграм-бот запущен.')
        telegram_api.RETRY_ON_ERROR = True
        telegram_api.SESSION_TIME_TO_LIVE = POLLING_SESSION_SECONDS
        self._bot.infinity_polling(
            skip_pending=True,
            timeout=POLLING_HTTP_TIMEOUT,
            long_polling_timeout=LONG_POLLING_TIMEOUT,
        )

    def _register_handlers(self) -> None:
        """Подключает команды и обработчик обычного текста."""
        self._bot.message_handler(commands=['start'])(self.on_start)
        self._bot.message_handler(commands=['forget'])(self.on_forget)
        self._bot.message_handler(content_types=['text'])(self.on_text)
        self._bot.message_handler(
            content_types=list(UNSUPPORTED_CONTENT_TYPES),
        )(self.on_unsupported)

    def on_start(self, message: telebot.types.Message) -> None:
        """Знакомится с пользователем и сохраняет его профиль.

        Args:
            message: Команда /start.
        """
        user = message.from_user
        if user is None:
            return
        self._try_remember(user, profile_text(user))
        self._reply(message, START_TEXT)

    def on_forget(self, message: telebot.types.Message) -> None:
        """Удаляет долговременную память пользователя и сеанс.

        Args:
            message: Команда /forget.
        """
        user = message.from_user
        if user is None:
            return
        try:
            self._memory.delete_documents(
                metadata_filter=user_memory_filter(user.id),
            )
        except PineconeManagerError as error:
            self._log_error('forget', user.id, error)
            self._reply(message, 'Не удалось очистить память. Попробуйте ещё раз.')
            return
        with self._lock:
            self._history.pop(user.id, None)
        self._reply(message, FORGET_TEXT)

    def on_unsupported(self, message: telebot.types.Message) -> None:
        """Отвечает, если прислали не текст.

        Args:
            message: Сообщение другого типа.
        """
        self._reply(message, UNSUPPORTED_TEXT)

    def on_text(self, message: telebot.types.Message) -> None:
        """Отвечает на реплику и записывает её в память пользователя.

        Args:
            message: Текстовое сообщение.
        """
        user = message.from_user
        raw_text = message.text or ''
        text = raw_text.strip()
        if user is None:
            return
        if text.startswith('/'):
            self._reply(message, UNKNOWN_COMMAND_TEXT)
            return
        if text == '':
            self._reply(message, EMPTY_TEXT)
            return
        self._bot.send_chat_action(message.chat.id, 'typing')
        memories = self._try_recall(user.id, text)
        with self._lock:
            history = list(self._history.get(user.id, []))
        try:
            answer = self._complete(
                build_chat_messages(format_memories(memories), history, text),
            )
        except TelegramBotError as error:
            self._log_error('chat', user.id, error)
            self._try_remember(user, text)
            self._reply(message, CHAT_ERROR_TEXT)
            return
        saved = self._try_remember(user, text)
        if not saved:
            answer = f'{answer}{MEMORY_ERROR_SUFFIX}'
        with self._lock:
            self._history[user.id] = trim_history(
                [
                    *history,
                    {'role': USER_ROLE, 'content': text},
                    {'role': ASSISTANT_ROLE, 'content': answer},
                ],
                MAX_HISTORY_MESSAGES,
            )
        self._reply(message, answer)

    def _try_recall(self, user_id: int, text: str) -> list[StoredDocument]:
        """Ищет память пользователя. Сбой поиска не обрывает диалог.

        Args:
            user_id: Идентификатор Telegram.
            text: Текст, по которому ищутся близкие факты.

        Returns:
            Найденные фрагменты или пустой список.
        """
        try:
            return self._memory.search_documents_by_text(
                text,
                field=MEMORY_TEXT_FIELD,
                top_k=MEMORY_TOP_K,
                score_type='embed',
                metadata_filter=user_memory_filter(user_id),
                include_fields=[MEMORY_TEXT_FIELD],
            )
        except PineconeManagerError as error:
            self._log_error('recall', user_id, error)
            return []

    def _try_remember(self, user: telebot.types.User, text: str) -> bool:
        """Записывает факт в документный индекс с дедупликацией по сходству.

        Индекс сам считает эмбеддинг поля ``chunk_text``. Ниже порога
        создаётся новый слот, точный повтор пропускается, вариация обновляет
        найденный документ.

        Args:
            user: Владелец факта.
            text: Что нужно сохранить.

        Returns:
            ``True``, если запись, обновление или пропуск прошли без ошибки.
        """
        try:
            nearest = self._memory.search_documents_by_text(
                text,
                field=MEMORY_TEXT_FIELD,
                top_k=1,
                score_type='embed',
                metadata_filter=user_memory_filter(user.id),
                include_fields=[MEMORY_TEXT_FIELD],
            )
            matched = nearest[0] if len(nearest) > 0 else None
            stored_text = None
            similarity = None
            if matched is not None:
                similarity = matched.score
                raw_text = matched.fields.get(MEMORY_TEXT_FIELD)
                if isinstance(raw_text, str):
                    stored_text = raw_text
            action = decide_memory_action(
                similarity,
                text,
                stored_text,
                require_cosine_threshold(COSINE_SIMILARITY_THRESHOLD),
            )
            if action == MEMORY_ACTION_SKIP:
                return True
            if action == MEMORY_ACTION_UPDATE and matched is not None:
                memory_id = matched.id
            else:
                memory_id = new_memory_id()
            document: dict[str, object] = {
                '_id': memory_id,
                MEMORY_TEXT_FIELD: text,
            }
            document.update(build_user_metadata(user))
            self._memory.upsert_schema_documents([document])
        except PineconeManagerError as error:
            self._log_error('remember', user.id, error)
            return False
        return True

    def _complete(self, messages: Sequence[Mapping[str, str]]) -> str:
        """Запрашивает ответ диалоговой модели.

        Args:
            messages: Подготовленный диалог.

        Returns:
            Текст ответа.

        Raises:
            TelegramBotError: Если модель не ответила.
        """
        try:
            response = self._chat_client.chat.completions.create(
                model=self._chat_model,
                messages=list(messages),
                temperature=CHAT_TEMPERATURE,
                max_completion_tokens=MAX_REPLY_TOKENS,
            )
        except Exception as error:
            raise TelegramBotError(
                code='chat_error',
                message='Диалоговая модель не ответила.',
                details={'error_type': type(error).__name__},
            ) from error
        choice = response.choices[0].message.content if response.choices else None
        if not isinstance(choice, str) or choice.strip() == '':
            return EMPTY_ANSWER_TEXT
        return choice.strip()

    def _reply(self, message: telebot.types.Message, text: str) -> None:
        """Отправляет ответ, разрезая его на допустимые части.

        Args:
            message: Исходное сообщение пользователя.
            text: Текст ответа.
        """
        parts = split_telegram_text(text)
        self._bot.reply_to(message, parts[0])
        for part in parts[1:]:
            self._bot.send_message(message.chat.id, part)

    def _log_error(self, operation: str, user_id: int, error: Exception) -> None:
        """Пишет в журнал тип сбоя без текста сообщения и секретов.

        Args:
            operation: Имя шага.
            user_id: Идентификатор пользователя.
            error: Пойманное исключение.
        """
        logger.error(
            'Ошибка бота: operation=%s user_id=%s error_type=%s code=%s',
            operation,
            user_id,
            type(error).__name__,
            getattr(error, 'code', ''),
        )


def main() -> None:
    """Читает настройки и запускает бота."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s %(message)s',
    )
    try:
        assistant = TelegramAssistant.from_env()
    except (TelegramBotError, PineconeManagerError) as error:
        logger.error(
            'Бот не запущен: error_type=%s code=%s message=%s',
            type(error).__name__,
            getattr(error, 'code', ''),
            getattr(error, 'message', error),
        )
        raise SystemExit(1) from error
    assistant.run()


if __name__ == '__main__':
    main()
