"""Чтение и запись данных в векторную базу Pinecone.

Класс ``PineconeManager`` закрывает два контура индекса:

* векторный — готовые эмбеддинги и тексты, которые модуль сам
  превращает в векторы через OpenAI;
* документный и встроенный — текст уходит в Pinecone, а эмбеддинг
  считает модель индекса.

Подготовка записей вынесена в чистые функции. Сетевые вызовы остаются
в методах класса.
"""

from __future__ import annotations

import logging
import math
import os
import re
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeVar, cast

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import (
    DenseVectorQuery,
    EmbedQuery,
    Pinecone,
    SparseEmbedQuery,
    TextQuery,
)

logger = logging.getLogger(__name__)

TItem = TypeVar('TItem')
TResult = TypeVar('TResult')

# Граница косинусного сходства для долговременной памяти чат-бота.
# Ниже границы сообщение — новая информация, его нужно запомнить.
# На границе и выше это дубликат или вариация: точный повтор пропускается,
# другая формулировка обновляет уже сохранённый слот.
# Допустимый диапазон: от -1 до 1. Выше число — реже срабатывает дедупликация.
COSINE_SIMILARITY_THRESHOLD = 0.85

# Сколько соседей возвращать, если вызывающий код не задал своё число.
DEFAULT_TOP_K = 5
# Сколько сохранённых фрагментов сравнивать с новым сообщением.
MEMORY_NEIGHBOR_COUNT = 5
MEMORY_ACTION_REMEMBER = 'remember'
MEMORY_ACTION_SKIP = 'skip'
MEMORY_ACTION_UPDATE = 'update'
# Пакет векторов в одном upsert. Дальше запрос упирается в лимит размера.
UPSERT_BATCH_SIZE = 100
# Пакет текстов в одном запросе эмбеддингов OpenAI.
EMBEDDING_BATCH_SIZE = 64
# Пакет JSON-документов. На индексе со встроенным эмбеддингом сервер
# принимает меньше записей, чем обычный векторный upsert.
SCHEMA_BATCH_SIZE = 50
# Поле, в которое кладётся исходный текст документа.
DEFAULT_TEXT_FIELD = 'text'
# Поле плотного вектора в документном индексе.
DEFAULT_VECTOR_FIELD = 'embedding'
# Модель OpenAI по умолчанию. Размерность должна совпадать с индексом.
DEFAULT_EMBEDDING_MODEL = 'text-embedding-3-small'
# Пустое имя пространства в Records API и Documents API так не передаётся.
INTEGRATED_DEFAULT_NAMESPACE = '__default__'
# Идентификатор Pinecone: 1–512 символов ASCII без NUL.
MAX_ID_LENGTH = 512
MAX_ERROR_MESSAGES = 5
ID_PATTERN = re.compile(r'^[\x01-\x7F]+$')
RESERVED_DOCUMENT_KEYS = frozenset({'_id', '_score', 'id', 'score'})
TEXT_SCORE_TYPES = frozenset({'embed', 'sparse_embed', 'text'})

PINECONE_API_KEY_ENV = 'PINECONE_API_KEY'
PINECONE_INDEX_NAME_ENV = 'PINECONE_INDEX_NAME'
PINECONE_NAMESPACE_ENV = 'PINECONE_NAMESPACE'
OPENAI_API_KEY_ENV = 'OPENAI_API_KEY'
OPENAI_BASE_URL_ENV = 'OPENAI_BASE_URL'
OPENAI_EMBEDDING_MODEL_ENV = 'OPENAI_EMBEDDING_MODEL'
OPENAI_EMBEDDING_DIMENSIONS_ENV = 'OPENAI_EMBEDDING_DIMENSIONS'

MetadataValue = str | int | float | bool | list[str]
TextScoreClause = EmbedQuery | SparseEmbedQuery | TextQuery


class PineconeManagerError(Exception):
    """Ошибка модуля: код, сообщение и подробности для вызывающего кода.

    Attributes:
        code: Короткий машинный код, например ``validation_error``.
        message: Текст для человека.
        details: Поля, по которым можно понять причину, без секретов.
    """

    def __init__(
        self,
        code: str,
        message: str,
        details: Mapping[str, object] | None = None,
    ) -> None:
        """Сохраняет код ошибки и необязательные подробности.

        Args:
            code: Машинный код ошибки.
            message: Сообщение, которое станет текстом исключения.
            details: Дополнительные поля. ``None`` заменяется пустым словарём.
        """
        self.code = code
        self.message = message
        self.details = dict(details or {})
        super().__init__(message)


@dataclass(frozen=True)
class VectorMatch:
    """Одно совпадение поиска по вектору."""

    id: str
    score: float
    metadata: dict[str, object]
    values: list[float]


@dataclass(frozen=True)
class TextMatch:
    """Одно совпадение поиска по тексту или по полям документа."""

    id: str
    score: float
    fields: dict[str, object]


@dataclass(frozen=True)
class StoredVector:
    """Вектор, прочитанный по идентификатору, без оценки близости."""

    id: str
    values: list[float]
    metadata: dict[str, object]


@dataclass(frozen=True)
class StoredDocument:
    """Документ, прочитанный по идентификатору или найденный поиском."""

    id: str
    fields: dict[str, object]
    score: float | None = None


@dataclass(frozen=True)
class IndexStats:
    """Сводка индекса: размерность, общий счётчик и счётчики пространств."""

    dimension: int | None
    total_vector_count: int
    namespaces: dict[str, int]


@dataclass(frozen=True)
class DocumentParts:
    """Текст документа, уже разобранный на идентификатор и метаданные."""

    vector_id: str
    text: str
    metadata: dict[str, MetadataValue]


@dataclass(frozen=True)
class MemoryWriteResult:
    """Чем закончилась попытка записать сообщение в долговременную память.

    Attributes:
        action: ``remember``, ``skip`` или ``update``.
        memory_id: Слот, в котором знание лежит после вызова.
        similarity: Сходство с ближайшим фрагментом. ``None``, если память пуста.
        matched_id: Идентификатор ближайшего фрагмента или ``None``.
    """

    action: str
    memory_id: str
    similarity: float | None
    matched_id: str | None


def read_env(name: str) -> str | None:
    """Возвращает значение переменной окружения без краевых пробелов.

    Args:
        name: Имя переменной.

    Returns:
        Строка или ``None``, если переменная отсутствует либо пустая.
    """
    raw = os.getenv(name)
    if raw is None:
        return None
    value = raw.strip()
    if value == '':
        return None
    return value


def clean_optional(value: str | None) -> str | None:
    """Убирает пробелы у необязательной строки.

    Args:
        value: Исходная строка или ``None``.

    Returns:
        Очищенная строка либо ``None``.
    """
    if value is None:
        return None
    cleaned = value.strip()
    if cleaned == '':
        return None
    return cleaned


def require_config(value: str | None, env_name: str) -> str:
    """Проверяет, что обязательная настройка задана.

    Args:
        value: Уже прочитанное значение.
        env_name: Имя переменной для текста ошибки.

    Returns:
        Ту же непустую строку.

    Raises:
        PineconeManagerError: Если значения нет.
    """
    if value is None:
        raise PineconeManagerError(
            code='config_error',
            message=f'Не задана настройка {env_name}.',
            details={'env': env_name},
        )
    return value


def read_optional_positive_int(name: str) -> int | None:
    """Читает положительное целое из окружения.

    Args:
        name: Имя переменной.

    Returns:
        Число или ``None``, если переменная не задана.

    Raises:
        PineconeManagerError: Если значение не является положительным целым.
    """
    raw = read_env(name)
    if raw is None:
        return None
    if not raw.isdigit() or int(raw) < 1:
        raise PineconeManagerError(
            code='config_error',
            message=f'{name} должна быть положительным целым числом.',
            details={'env': name, 'value': raw},
        )
    return int(raw)


def require_identifier(value: object, field_name: str) -> str:
    """Проверяет идентификатор записи Pinecone.

    Args:
        value: Кандидат в идентификатор.
        field_name: Имя поля для текста ошибки.

    Returns:
        Идентификатор без краевых пробелов.

    Raises:
        PineconeManagerError: Если значение не является допустимым id.
    """
    if not isinstance(value, str):
        raise PineconeManagerError(
            code='validation_error',
            message=f'Поле «{field_name}» должно быть строкой.',
            details={'field': field_name},
        )
    cleaned = value.strip()
    if cleaned == '' or len(cleaned) > MAX_ID_LENGTH:
        raise PineconeManagerError(
            code='validation_error',
            message=(
                f'Поле «{field_name}» должно содержать от 1 до '
                f'{MAX_ID_LENGTH} символов.'
            ),
            details={'field': field_name},
        )
    if ID_PATTERN.fullmatch(cleaned) is None:
        raise PineconeManagerError(
            code='validation_error',
            message=(
                f'Поле «{field_name}» должно состоять из ASCII-символов '
                'без нулевого байта.'
            ),
            details={'field': field_name},
        )
    return cleaned


def require_text(value: object, field_name: str) -> str:
    """Проверяет, что текст для эмбеддинга не пустой.

    Args:
        value: Кандидат в текст.
        field_name: Имя поля для текста ошибки.

    Returns:
        Текст без краевых пробелов.

    Raises:
        PineconeManagerError: Если это не непустая строка.
    """
    if not isinstance(value, str):
        raise PineconeManagerError(
            code='validation_error',
            message=f'Поле «{field_name}» должно быть строкой.',
            details={'field': field_name},
        )
    cleaned = value.strip()
    if cleaned == '':
        raise PineconeManagerError(
            code='validation_error',
            message=f'Поле «{field_name}» не должно быть пустым.',
            details={'field': field_name},
        )
    return cleaned


def require_positive_int(value: object, field_name: str) -> int:
    """Проверяет положительное целое, не путая его с логическим значением.

    Args:
        value: Проверяемое значение.
        field_name: Имя поля для текста ошибки.

    Returns:
        То же число.

    Raises:
        PineconeManagerError: Если число не является положительным целым.
    """
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise PineconeManagerError(
            code='validation_error',
            message=f'Поле «{field_name}» должно быть положительным целым.',
            details={'field': field_name},
        )
    return value


def require_non_empty(items: Sequence[object], field_name: str) -> None:
    """Запрещает пустой список на входе операции записи или чтения.

    Args:
        items: Проверяемая последовательность.
        field_name: Имя списка для текста ошибки.

    Raises:
        PineconeManagerError: Если список пуст.
    """
    if len(items) == 0:
        raise PineconeManagerError(
            code='validation_error',
            message=f'Список «{field_name}» не должен быть пустым.',
            details={'field': field_name},
        )


def ensure_unique_ids(vector_ids: Sequence[str]) -> None:
    """Проверяет, что идентификаторы внутри одной записи не повторяются.

    Args:
        vector_ids: Идентификаторы в порядке передачи.

    Raises:
        PineconeManagerError: Если хотя бы один идентификатор встретился дважды.
    """
    seen: set[str] = set()
    duplicates: list[str] = []
    for vector_id in vector_ids:
        if vector_id in seen and vector_id not in duplicates:
            duplicates.append(vector_id)
        seen.add(vector_id)
    if duplicates:
        raise PineconeManagerError(
            code='validation_error',
            message='Идентификаторы записей должны быть уникальными.',
            details={'duplicate_ids': duplicates},
        )


def split_batches(items: Sequence[TItem], batch_size: int) -> list[list[TItem]]:
    """Делит последовательность на пакеты фиксированного размера.

    Args:
        items: Исходные элементы.
        batch_size: Размер одного пакета.

    Returns:
        Список пакетов. Пустой вход даёт пустой список.
    """
    size = require_positive_int(batch_size, 'batch_size')
    return [
        list(items[start:start + size])
        for start in range(0, len(items), size)
    ]


def require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    """Проверяет, что значение является словарём.

    Args:
        value: Проверяемое значение.
        field_name: Имя поля для текста ошибки.

    Returns:
        Тот же словарь.

    Raises:
        PineconeManagerError: Если это не словарь.
    """
    if not isinstance(value, Mapping):
        raise PineconeManagerError(
            code='validation_error',
            message=f'Поле «{field_name}» должно быть словарём.',
            details={'field': field_name},
        )
    return cast(Mapping[str, object], value)


def copy_filter(
    metadata_filter: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Копирует фильтр метаданных, не изменяя словарь вызывающего кода.

    Args:
        metadata_filter: Фильтр Pinecone или ``None``.

    Returns:
        Поверхностная копия либо ``None``.

    Raises:
        PineconeManagerError: Если передан пустой фильтр.
    """
    if metadata_filter is None:
        return None
    if len(metadata_filter) == 0:
        raise PineconeManagerError(
            code='validation_error',
            message='Фильтр метаданных не должен быть пустым.',
            details={'field': 'metadata_filter'},
        )
    return dict(metadata_filter)


def normalize_metadata_value(key: str, value: object) -> MetadataValue:
    """Приводит одно значение метаданных к типу, который принимает Pinecone.

    Args:
        key: Имя поля, попадёт в текст ошибки.
        value: Проверяемое значение.

    Returns:
        Строка, число, логическое значение или список строк.

    Raises:
        PineconeManagerError: Если тип метаданных не поддерживается.
    """
    if isinstance(value, bool) or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        items = list(value)
        if all(isinstance(item, str) for item in items):
            return cast(list[str], items)
    raise PineconeManagerError(
        code='validation_error',
        message=(
            'Метаданные Pinecone принимают строку, число, '
            'логическое значение или список строк.'
        ),
        details={'field': key, 'value_type': type(value).__name__},
    )


def normalize_metadata(
    metadata: Mapping[str, object] | None,
) -> dict[str, MetadataValue]:
    """Проверяет словарь метаданных целиком.

    Args:
        metadata: Метаданные или ``None``.

    Returns:
        Новый словарь с допустимыми значениями.
    """
    if metadata is None:
        return {}
    normalized: dict[str, MetadataValue] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or key.strip() == '':
            raise PineconeManagerError(
                code='validation_error',
                message='Имена полей метаданных должны быть непустыми строками.',
                details={'field': str(key)},
            )
        normalized[key] = normalize_metadata_value(key, value)
    return normalized


def normalize_vector(values: Sequence[object]) -> list[float]:
    """Проверяет плотный вектор и приводит компоненты к ``float``.

    Args:
        values: Компоненты вектора.

    Returns:
        Новый список чисел.

    Raises:
        PineconeManagerError: Если вектор пуст или содержит нечисло.
    """
    if isinstance(values, (str, bytes)) or len(values) == 0:
        raise PineconeManagerError(
            code='validation_error',
            message='Вектор должен быть непустым списком чисел.',
            details={'field': 'values'},
        )
    normalized: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PineconeManagerError(
                code='validation_error',
                message='Вектор должен состоять из чисел.',
                details={'field': 'values', 'index': index},
            )
        normalized.append(float(value))
    return normalized


def require_cosine_threshold(value: object) -> float:
    """Проверяет границу косинусного сходства.

    Args:
        value: Число из глобальной настройки.

    Returns:
        Граница в диапазоне от -1 до 1.

    Raises:
        PineconeManagerError: Если значение не число или лежит вне диапазона.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PineconeManagerError(
            code='config_error',
            message='COSINE_SIMILARITY_THRESHOLD должна быть числом от -1 до 1.',
            details={'value': value},
        )
    threshold = float(value)
    if threshold < -1.0 or threshold > 1.0:
        raise PineconeManagerError(
            code='config_error',
            message='COSINE_SIMILARITY_THRESHOLD должна быть числом от -1 до 1.',
            details={'value': threshold},
        )
    return threshold


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Считает косинусное сходство двух векторов одной размерности.

    Нулевой вектор даёт сходство 0: сравнивать его направление не с чем.

    Args:
        left: Первый вектор.
        right: Второй вектор.

    Returns:
        Сходство от -1 до 1.

    Raises:
        PineconeManagerError: Если длины векторов различаются или вектор пуст.
    """
    if len(left) == 0 or len(left) != len(right):
        raise PineconeManagerError(
            code='validation_error',
            message='Для косинусного сходства нужны векторы одной ненулевой длины.',
            details={'left': len(left), 'right': len(right)},
        )
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        dot += left_value * right_value
        left_norm += left_value * left_value
        right_norm += right_value * right_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))


def new_memory_id() -> str:
    """Создаёт идентификатор нового слота памяти.

    Returns:
        ASCII-идентификатор вида ``mem-`` и шестнадцатеричный uuid.
    """
    return f'mem-{uuid.uuid4().hex}'


def stored_memory_text(
    metadata: Mapping[str, object],
    text_field: str,
) -> str | None:
    """Достаёт текст сохранённого фрагмента из метаданных.

    Args:
        metadata: Метаданные найденного слота.
        text_field: Имя текстового поля.

    Returns:
        Текст или ``None``, если поля нет или оно не строка.
    """
    value = metadata.get(text_field)
    if isinstance(value, str):
        return value
    return None


def decide_memory_action(
    similarity: float | None,
    current_text: str,
    stored_text: str | None,
    threshold: float,
) -> str:
    """Решает, запоминать сообщение, пропустить его или обновить слот.

    Граница включается в «высокое» сходство. Пустая память и сходство
    ниже границы считаются новой информацией.

    Args:
        similarity: Сходство с ближайшим фрагментом или ``None``.
        current_text: Текст текущего сообщения.
        stored_text: Текст ближайшего фрагмента или ``None``.
        threshold: Граница из ``COSINE_SIMILARITY_THRESHOLD``.

    Returns:
        ``remember``, ``skip`` или ``update``.
    """
    if similarity is None or similarity < threshold:
        return MEMORY_ACTION_REMEMBER
    if stored_text is not None and stored_text.strip().casefold() == current_text.casefold():
        return MEMORY_ACTION_SKIP
    return MEMORY_ACTION_UPDATE


def closest_memory(
    vector: Sequence[float],
    matches: Sequence[VectorMatch],
) -> tuple[VectorMatch, float] | None:
    """Выбирает сохранённый фрагмент с наибольшим косинусным сходством.

    Args:
        vector: Вектор текущего сообщения.
        matches: Кандидаты, у которых уже заполнены ``values``.

    Returns:
        Пара из совпадения и сходства. ``None``, если сравнивать не с чем.
    """
    best_match: VectorMatch | None = None
    best_score = -1.0
    found = False
    for match in matches:
        if len(match.values) != len(vector):
            continue
        score = cosine_similarity(vector, match.values)
        if not found or score > best_score:
            best_match = match
            best_score = score
            found = True
    if best_match is None:
        return None
    return best_match, best_score


def merge_memory_metadata(
    stored: Mapping[str, object],
    incoming: Mapping[str, object] | None,
    text_field: str,
    text: str,
    store_text: bool,
) -> dict[str, object]:
    """Собирает метаданные слота: старые поля, новые поля и текст.

    Args:
        stored: Метаданные уже записанного слота.
        incoming: Поля текущего сообщения.
        text_field: Имя поля текста.
        text: Текст, который нужно сохранить.
        store_text: Класть ли текст в метаданные.

    Returns:
        Новый словарь метаданных. Исходные словари не меняются.
    """
    merged: dict[str, object] = dict(stored)
    if incoming is not None:
        merged.update(incoming)
    if store_text:
        merged[text_field] = text
    return merged


def normalize_sparse_vector(
    sparse_values: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Проверяет разреженный вектор из индексов и значений.

    Args:
        sparse_values: Словарь с ключами ``indices`` и ``values`` либо ``None``.

    Returns:
        Нормализованный словарь или ``None``, если разреженной части нет.
    """
    if sparse_values is None:
        return None
    if 'indices' not in sparse_values or 'values' not in sparse_values:
        raise PineconeManagerError(
            code='validation_error',
            message='Разреженный вектор должен содержать indices и values.',
            details={'field': 'sparse_values'},
        )
    raw_indices = sparse_values['indices']
    raw_values = sparse_values['values']
    if (
        not isinstance(raw_indices, Sequence)
        or isinstance(raw_indices, (str, bytes))
        or not isinstance(raw_values, Sequence)
        or isinstance(raw_values, (str, bytes))
    ):
        raise PineconeManagerError(
            code='validation_error',
            message='indices и values разреженного вектора должны быть списками.',
            details={'field': 'sparse_values'},
        )
    if len(raw_indices) == 0 or len(raw_indices) != len(raw_values):
        raise PineconeManagerError(
            code='validation_error',
            message='Длины indices и values разреженного вектора должны совпадать.',
            details={'field': 'sparse_values'},
        )
    indices: list[int] = []
    for position, index in enumerate(raw_indices):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise PineconeManagerError(
                code='validation_error',
                message='Индексы разреженного вектора должны быть неотрицательными.',
                details={'field': 'sparse_values', 'index': position},
            )
        indices.append(index)
    return {'indices': indices, 'values': normalize_vector(list(raw_values))}


def build_vector_payload(
    vector_id: str,
    values: Sequence[object],
    metadata: Mapping[str, object] | None = None,
    sparse_values: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Собирает тело одного вектора для upsert.

    Args:
        vector_id: Идентификатор записи.
        values: Плотный вектор.
        metadata: Метаданные. Пустой словарь в запрос не попадает.
        sparse_values: Необязательная разреженная часть.

    Returns:
        Словарь с ключами ``id``, ``values`` и необязательными полями.
    """
    payload: dict[str, object] = {
        'id': require_identifier(vector_id, 'id'),
        'values': normalize_vector(values),
    }
    normalized_metadata = normalize_metadata(metadata)
    if normalized_metadata:
        payload['metadata'] = normalized_metadata
    normalized_sparse = normalize_sparse_vector(sparse_values)
    if normalized_sparse is not None:
        payload['sparse_values'] = normalized_sparse
    return payload


def parse_vector_item(item: Mapping[str, object]) -> dict[str, object]:
    """Превращает словарь вызывающего кода в проверенное тело вектора.

    Args:
        item: Словарь с ``id``, ``values`` и необязательными полями.

    Returns:
        Тело для ``upsert``.
    """
    if 'id' not in item or 'values' not in item:
        raise PineconeManagerError(
            code='validation_error',
            message='Вектор должен содержать поля id и values.',
            details={'fields': [str(key) for key in item]},
        )
    values = item['values']
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise PineconeManagerError(
            code='validation_error',
            message='Поле values должно быть списком чисел.',
            details={'field': 'values'},
        )
    metadata = item.get('metadata')
    sparse_values = item.get('sparse_values')
    metadata_mapping = None
    sparse_mapping = None
    if metadata is not None:
        metadata_mapping = require_mapping(metadata, 'metadata')
    if sparse_values is not None:
        sparse_mapping = require_mapping(sparse_values, 'sparse_values')
    return build_vector_payload(
        require_identifier(item['id'], 'id'),
        list(values),
        metadata_mapping,
        sparse_mapping,
    )


def extract_document_parts(
    document: Mapping[str, object],
    id_field: str,
    text_field: str,
    store_text: bool,
) -> DocumentParts:
    """Достаёт из документа идентификатор, текст и метаданные.

    Args:
        document: Исходный документ.
        id_field: Имя поля с идентификатором.
        text_field: Имя поля с текстом.
        store_text: Класть ли текст в метаданные вектора.

    Returns:
        Разобранные части документа.
    """
    if id_field not in document:
        raise PineconeManagerError(
            code='validation_error',
            message=f'В документе нет поля «{id_field}».',
            details={'field': id_field},
        )
    if text_field not in document:
        raise PineconeManagerError(
            code='validation_error',
            message=f'В документе нет поля «{text_field}».',
            details={'field': text_field},
        )
    vector_id = require_identifier(document[id_field], id_field)
    text = require_text(document[text_field], text_field)
    metadata_source: dict[str, object] = {}
    for key, value in document.items():
        if key == id_field:
            continue
        if key == text_field and not store_text:
            continue
        metadata_source[key] = value
    if store_text:
        metadata_source[text_field] = text
    return DocumentParts(
        vector_id=vector_id,
        text=text,
        metadata=normalize_metadata(metadata_source),
    )


def prepare_record_id(record: Mapping[str, object]) -> dict[str, object]:
    """Оставляет у записи один идентификатор в поле ``_id``.

    Если переданы и ``id``, и ``_id``, сохраняется ``_id``.

    Args:
        record: Исходная запись.

    Returns:
        Копия записи с проверенным ``_id``.
    """
    prepared = dict(record)
    if '_id' not in prepared and 'id' in prepared:
        prepared['_id'] = prepared.pop('id')
    prepared['_id'] = require_identifier(prepared.get('_id'), '_id')
    return prepared


def prepare_text_record(record: Mapping[str, object]) -> dict[str, object]:
    """Готовит текстовую запись для индекса со встроенным эмбеддингом.

    Args:
        record: Запись с идентификатором и хотя бы одним текстовым полем.

    Returns:
        Запись с полем ``_id``.
    """
    prepared = prepare_record_id(record)
    if len(prepared) < 2:
        raise PineconeManagerError(
            code='validation_error',
            message='Текстовая запись должна содержать поля помимо идентификатора.',
            details={'id': prepared.get('_id')},
        )
    return prepared


def build_text_score(score_type: str, text: str, field: str) -> TextScoreClause:
    """Собирает условие документного поиска по тексту.

    Args:
        score_type: ``embed``, ``sparse_embed`` или ``text``.
        text: Текст запроса.
        field: Поле схемы, по которому считается оценка.

    Returns:
        Условие ``score_by`` для Documents API.
    """
    if score_type == 'embed':
        return EmbedQuery(query=text, field=field)
    if score_type == 'sparse_embed':
        return SparseEmbedQuery(query=text, field=field)
    if score_type == 'text':
        return TextQuery(query=text, field=field)
    raise PineconeManagerError(
        code='validation_error',
        message='Тип текстовой оценки должен быть embed, sparse_embed или text.',
        details={'score_type': score_type, 'allowed': sorted(TEXT_SCORE_TYPES)},
    )


def resolve_include_fields(include_fields: Sequence[str] | None) -> list[str]:
    """Выбирает поля документа, которые нужно вернуть из поиска.

    ``None`` означает все поля. Пустой список оставляет только идентификатор
    и оценку: так устроен Documents API.

    Args:
        include_fields: Имена полей или ``None``.

    Returns:
        Список имён для аргумента ``include_fields``.
    """
    if include_fields is None:
        return ['*']
    return [require_text(name, 'include_fields') for name in include_fields]


def read_attr(source: object, name: str, default: object = None) -> object:
    """Читает поле и у объекта SDK, и у словаря.

    Args:
        source: Ответ SDK или словарь.
        name: Имя поля.
        default: Значение, если поля нет.

    Returns:
        Найденное значение или ``default``.
    """
    if isinstance(source, Mapping):
        mapping = cast(Mapping[str, object], source)
        return mapping.get(name, default)
    return getattr(source, name, default)


def as_int(value: object, default: int = 0) -> int:
    """Приводит число к ``int``, не принимая логические значения за числа.

    Args:
        value: Исходное значение.
        default: Подстановка, если значение не число.

    Returns:
        Целое число.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return int(value)


def as_float(value: object, default: float = 0.0) -> float:
    """Приводит число к ``float``.

    Args:
        value: Исходное значение.
        default: Подстановка, если значение не число.

    Returns:
        Число с плавающей точкой.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def as_float_list(value: object) -> list[float]:
    """Собирает список чисел из ответа SDK.

    Args:
        value: Список либо иное значение.

    Returns:
        Список ``float``. Несписок даёт пустой результат.

    Raises:
        PineconeManagerError: Если элемент списка не является числом.
    """
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    numbers: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise PineconeManagerError(
                code='response_error',
                message='В ответе Pinecone вектор содержит нечисло.',
                details={'index': index},
            )
        numbers.append(float(item))
    return numbers


def mapping_to_dict(value: object) -> dict[str, object]:
    """Превращает словарь или модель с ``to_dict`` в обычный словарь.

    Args:
        value: Метаданные, документ или ``None``.

    Returns:
        Новый словарь. Для неподдерживаемого значения возвращает пустой.
    """
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    to_dict = getattr(value, 'to_dict', None)
    if not callable(to_dict):
        return {}
    dumped = to_dict()
    if not isinstance(dumped, Mapping):
        return {}
    return {str(key): item for key, item in dumped.items()}


def document_fields(document: object) -> dict[str, object]:
    """Достаёт пользовательские поля совпадения, без id и score.

    Args:
        document: ``Hit``, ``Document`` или словарь.

    Returns:
        Поля записи.
    """
    raw_fields = read_attr(document, 'fields', None)
    if isinstance(raw_fields, Mapping):
        return {str(key): value for key, value in raw_fields.items()}
    dumped = mapping_to_dict(document)
    return {
        key: value
        for key, value in dumped.items()
        if key not in RESERVED_DOCUMENT_KEYS
    }


def read_match_id(item: object) -> str:
    """Читает идентификатор совпадения из ``id`` или ``_id``.

    Args:
        item: Элемент ответа SDK.

    Returns:
        Идентификатор или пустая строка.
    """
    for name in ('id', '_id'):
        value = read_attr(item, name, None)
        if isinstance(value, str) and value != '':
            return value
    return ''


def read_optional_score(item: object) -> float | None:
    """Читает оценку, если ответ её содержит.

    Args:
        item: Элемент ответа SDK.

    Returns:
        Оценка или ``None`` для выборки без ранжирования.
    """
    for name in ('score', '_score'):
        value = read_attr(item, name, None)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return None


def iter_hits(response: object) -> list[object]:
    """Достаёт список совпадений и из Records API, и из Documents API.

    Args:
        response: Ответ ``search`` или ``documents.search``.

    Returns:
        Совпадения в порядке, который вернул сервер.
    """
    result = read_attr(response, 'result', None)
    if result is not None:
        hits = read_attr(result, 'hits', None)
        if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes)):
            return list(hits)
    matches = read_attr(response, 'matches', None)
    if isinstance(matches, Sequence) and not isinstance(matches, (str, bytes)):
        return list(matches)
    return []


def normalize_vector_matches(response: object) -> list[VectorMatch]:
    """Превращает ответ ``query`` в список ``VectorMatch``.

    Args:
        response: Ответ векторного запроса.

    Returns:
        Совпадения от наиболее близкого к наименее близкому.
    """
    raw_matches = read_attr(response, 'matches', [])
    if not isinstance(raw_matches, Sequence) or isinstance(raw_matches, (str, bytes)):
        return []
    matches: list[VectorMatch] = []
    for item in raw_matches:
        matches.append(
            VectorMatch(
                id=read_match_id(item),
                score=as_float(read_attr(item, 'score', 0.0)),
                metadata=mapping_to_dict(read_attr(item, 'metadata', None)),
                values=as_float_list(read_attr(item, 'values', [])),
            )
        )
    return matches


def normalize_text_matches(response: object) -> list[TextMatch]:
    """Превращает ответ текстового поиска в список ``TextMatch``.

    Args:
        response: Ответ ``search`` или ``documents.search``.

    Returns:
        Совпадения с полями записи.
    """
    matches: list[TextMatch] = []
    for item in iter_hits(response):
        score = read_optional_score(item)
        matches.append(
            TextMatch(
                id=read_match_id(item),
                score=0.0 if score is None else score,
                fields=document_fields(item),
            )
        )
    return matches


def normalize_stored_vectors(
    response: object,
    requested_ids: Sequence[str],
) -> list[StoredVector]:
    """Собирает векторы выборки по id в порядке запроса.

    Отсутствующие идентификаторы пропускаются: Pinecone их не считает ошибкой.

    Args:
        response: Ответ ``fetch``.
        requested_ids: Идентификаторы в исходном порядке.

    Returns:
        Найденные векторы.
    """
    raw_vectors = read_attr(response, 'vectors', {})
    if not isinstance(raw_vectors, Mapping):
        return []
    vectors = cast(Mapping[str, object], raw_vectors)
    stored: list[StoredVector] = []
    for vector_id in requested_ids:
        raw = vectors.get(vector_id)
        if raw is None:
            continue
        stored.append(
            StoredVector(
                id=read_match_id(raw) or vector_id,
                values=as_float_list(read_attr(raw, 'values', [])),
                metadata=mapping_to_dict(read_attr(raw, 'metadata', None)),
            )
        )
    return stored


def normalize_stored_documents(response: object) -> list[StoredDocument]:
    """Превращает выборку документов в список ``StoredDocument``.

    Args:
        response: Ответ ``documents.fetch`` или поиск документов.

    Returns:
        Документы с пользовательскими полями.
    """
    raw_documents = read_attr(response, 'documents', None)
    if isinstance(raw_documents, Mapping):
        pairs = list(raw_documents.items())
    else:
        pairs = [(read_match_id(item), item) for item in iter_hits(response)]
    stored: list[StoredDocument] = []
    for key, document in pairs:
        document_id = read_match_id(document) or str(key)
        stored.append(
            StoredDocument(
                id=document_id,
                fields=document_fields(document),
                score=read_optional_score(document),
            )
        )
    return stored


def accepted_count(response: object, fallback: int) -> int:
    """Достаёт число принятых записей из разных форм ответа upsert.

    Args:
        response: Ответ SDK.
        fallback: Число, если счётчик в ответе не найден.

    Returns:
        Сколько записей сервер принял.
    """
    for name in ('upserted_count', 'record_count', 'successful_item_count'):
        value = read_attr(response, name, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return fallback


def error_messages(response: object) -> list[str]:
    """Собирает короткие тексты ошибок пакета, без тел записей.

    Args:
        response: Ответ пакетной записи.

    Returns:
        Не больше ``MAX_ERROR_MESSAGES`` строк.
    """
    raw = read_attr(response, 'errors', None)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item) for item in list(raw)[:MAX_ERROR_MESSAGES]]
    return [str(raw)]


def failed_item_count(response: object) -> int:
    """Возвращает число незаписанных элементов пакетного ответа.

    Args:
        response: Ответ SDK.

    Returns:
        Число ошибок. Отсутствие поля означает ноль.
    """
    if bool(read_attr(response, 'has_errors', False)):
        reported = as_int(read_attr(response, 'failed_item_count', 0))
        if reported > 0:
            return reported
        return 1
    return as_int(read_attr(response, 'failed_item_count', 0))


class PineconeManager:
    """Управляет чтением и записью одного индекса Pinecone.

    Ключи берутся из аргументов или из окружения. Файл ``.env``
    подхватывается при создании объекта и не перезаписывает уже
    заданные переменные.

    Запись:

    * ``remember`` — сообщение в долговременную память с проверкой сходства;
    * ``upsert_vector`` / ``upsert_vectors`` — готовые векторы;
    * ``upsert_document`` / ``upsert_documents`` — тексты, эмбеддинг OpenAI;
    * ``upsert_text_records`` — тексты для индекса со встроенной моделью;
    * ``upsert_schema_documents`` — JSON-документы документного индекса.

    Чтение:

    * ``query_by_vector`` / ``query_by_vector_id`` — по вектору;
    * ``query_by_text`` — текст считается в вектор через OpenAI;
    * ``search_by_text`` — текст считает модель встроенного индекса;
    * ``search_documents_by_vector`` / ``search_documents_by_text`` —
      чтение документного индекса.
    """

    def __init__(
        self,
        api_key: str | None = None,
        index_name: str | None = None,
        namespace: str | None = None,
        openai_api_key: str | None = None,
        openai_base_url: str | None = None,
        embedding_model: str | None = None,
        embedding_dimensions: int | None = None,
    ) -> None:
        """Открывает клиент Pinecone и запоминает настройки эмбеддинга.

        Args:
            api_key: Ключ Pinecone. По умолчанию ``PINECONE_API_KEY``.
            index_name: Имя индекса. По умолчанию ``PINECONE_INDEX_NAME``.
            namespace: Пространство по умолчанию. Пустая строка — основное
                пространство векторного индекса.
            openai_api_key: Ключ OpenAI для текстовых операций.
            openai_base_url: Необязательный базовый URL OpenAI.
            embedding_model: Модель эмбеддингов OpenAI.
            embedding_dimensions: Укороченная размерность для моделей,
                которые её поддерживают. ``None`` оставляет размер модели.

        Raises:
            PineconeManagerError: Если не хватает настроек или индекс
                не открывается.
        """
        load_dotenv()
        resolved_key = require_config(
            clean_optional(api_key) or read_env(PINECONE_API_KEY_ENV),
            PINECONE_API_KEY_ENV,
        )
        resolved_index = require_config(
            clean_optional(index_name) or read_env(PINECONE_INDEX_NAME_ENV),
            PINECONE_INDEX_NAME_ENV,
        )
        if namespace is None:
            self._default_namespace = read_env(PINECONE_NAMESPACE_ENV) or ''
        else:
            self._default_namespace = namespace
        self._index_name = resolved_index
        self._openai_api_key = (
            clean_optional(openai_api_key) or read_env(OPENAI_API_KEY_ENV)
        )
        self._openai_base_url = (
            clean_optional(openai_base_url) or read_env(OPENAI_BASE_URL_ENV)
        )
        self._embedding_model = require_config(
            clean_optional(embedding_model)
            or read_env(OPENAI_EMBEDDING_MODEL_ENV)
            or DEFAULT_EMBEDDING_MODEL,
            OPENAI_EMBEDDING_MODEL_ENV,
        )
        if embedding_dimensions is None:
            self._embedding_dimensions = read_optional_positive_int(
                OPENAI_EMBEDDING_DIMENSIONS_ENV,
            )
        else:
            self._embedding_dimensions = require_positive_int(
                embedding_dimensions,
                'embedding_dimensions',
            )
        self._openai: OpenAI | None = None
        self._closed = False
        self._index = self._open_index(resolved_key, resolved_index)

    def upsert_vector(
        self,
        vector_id: str,
        values: Sequence[float],
        metadata: Mapping[str, object] | None = None,
        sparse_values: Mapping[str, object] | None = None,
        namespace: str | None = None,
    ) -> int:
        """Записывает один готовый вектор.

        Повторная запись с тем же идентификатором заменяет вектор целиком.

        Args:
            vector_id: Идентификатор записи.
            values: Плотный вектор размерности индекса.
            metadata: Метаданные записи.
            sparse_values: Разреженная часть для гибридного индекса.
            namespace: Пространство. ``None`` — пространство по умолчанию.

        Returns:
            Число принятых векторов.
        """
        payload = build_vector_payload(
            vector_id,
            list(values),
            metadata,
            sparse_values,
        )
        return self.upsert_vectors([payload], namespace=namespace)

    def upsert_vectors(
        self,
        vectors: Sequence[Mapping[str, object]],
        namespace: str | None = None,
        batch_size: int = UPSERT_BATCH_SIZE,
    ) -> int:
        """Записывает пакет готовых векторов.

        Args:
            vectors: Словари с полями ``id``, ``values`` и необязательно
                ``metadata`` и ``sparse_values``.
            namespace: Пространство. ``None`` — пространство по умолчанию.
            batch_size: Сколько векторов отправлять в одном запросе.

        Returns:
            Число принятых векторов.

        Raises:
            PineconeManagerError: Если пакет принят частично или запрос отклонён.
        """
        require_non_empty(list(vectors), 'vectors')
        payloads = [parse_vector_item(item) for item in vectors]
        ensure_unique_ids([str(item['id']) for item in payloads])
        namespace_name = self._namespace(namespace)
        size = require_positive_int(batch_size, 'batch_size')

        def action() -> object:
            return self._index.upsert(
                vectors=payloads,
                namespace=namespace_name,
                batch_size=size,
                show_progress=False,
            )

        response = self._call('upsert_vectors', action)
        return self._require_full_write(response, len(payloads), 'upsert_vectors')

    def upsert_document(
        self,
        document_id: str,
        text: str,
        metadata: Mapping[str, object] | None = None,
        namespace: str | None = None,
        text_field: str = DEFAULT_TEXT_FIELD,
        store_text: bool = True,
    ) -> int:
        """Считает эмбеддинг одного текста и записывает его как вектор.

        Args:
            document_id: Идентификатор документа.
            text: Текст, который пойдёт в модель OpenAI.
            metadata: Дополнительные метаданные.
            namespace: Пространство. ``None`` — пространство по умолчанию.
            text_field: Имя поля, под которым текст ляжет в метаданные.
            store_text: Сохранять ли исходный текст рядом с вектором.

        Returns:
            Число принятых векторов.
        """
        # Явные id и текст остаются за аргументами метода, а не за метаданными.
        document: dict[str, object] = {}
        if metadata is not None:
            document.update(metadata)
        document['id'] = document_id
        document[text_field] = text
        return self.upsert_documents(
            [document],
            namespace=namespace,
            text_field=text_field,
            store_text=store_text,
        )

    def upsert_documents(
        self,
        documents: Sequence[Mapping[str, object]],
        namespace: str | None = None,
        id_field: str = 'id',
        text_field: str = DEFAULT_TEXT_FIELD,
        store_text: bool = True,
        batch_size: int = UPSERT_BATCH_SIZE,
    ) -> int:
        """Считает эмбеддинги текстов через OpenAI и записывает векторы.

        Размерность модели должна совпадать с размерностью индекса.
        Для моделей семейства ``text-embedding-3`` её можно укоротить
        параметром ``embedding_dimensions``.

        Args:
            documents: Документы с полями идентификатора и текста.
            namespace: Пространство. ``None`` — пространство по умолчанию.
            id_field: Имя поля идентификатора.
            text_field: Имя текстового поля.
            store_text: Класть ли текст в метаданные.
            batch_size: Размер пакета upsert после эмбеддинга.

        Returns:
            Число принятых векторов.
        """
        require_non_empty(list(documents), 'documents')
        if id_field == text_field:
            raise PineconeManagerError(
                code='validation_error',
                message='Поля идентификатора и текста должны называться по-разному.',
                details={'field': id_field},
            )
        parts = [
            extract_document_parts(document, id_field, text_field, store_text)
            for document in documents
        ]
        ensure_unique_ids([part.vector_id for part in parts])
        vectors = self._embed_texts([part.text for part in parts])
        payloads = [
            build_vector_payload(part.vector_id, vector, part.metadata)
            for part, vector in zip(parts, vectors, strict=True)
        ]
        return self.upsert_vectors(
            payloads,
            namespace=namespace,
            batch_size=batch_size,
        )

    def remember(
        self,
        text: str,
        memory_id: str | None = None,
        metadata: Mapping[str, object] | None = None,
        namespace: str | None = None,
        text_field: str = DEFAULT_TEXT_FIELD,
        store_text: bool = True,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> MemoryWriteResult:
        """Кладёт сообщение в долговременную память, если это новое знание.

        Сначала считается эмбеддинг текста и косинусное сходство с уже
        сохранёнными фрагментами. Граница задаётся глобальной переменной
        ``COSINE_SIMILARITY_THRESHOLD``.

        * Ниже границы — новая информация, создаётся новый слот.
        * На границе и выше при том же тексте — дубликат, запись пропускается.
        * На границе и выше при другой формулировке — обновляется найденный слот.

        Args:
            text: Сообщение, которое бот хочет запомнить.
            memory_id: Идентификатор нового слота. Для обновления не используется:
                остаётся id уже найденного фрагмента. ``None`` — id создаётся сам.
            metadata: Дополнительные метаданные слота.
            namespace: Пространство. ``None`` — пространство по умолчанию.
            text_field: Поле метаданных с исходным текстом.
            store_text: Сохранять ли текст в метаданных. Без текста повтор
                нельзя отличить от вариации, поэтому высокое сходство обновляет слот.
            metadata_filter: Ограничение, среди каких записей искать похожие.

        Returns:
            Решение: запомнить, пропустить или обновить, и id итогового слота.
        """
        cleaned = require_text(text, 'text')
        field_name = require_text(text_field, 'text_field')
        threshold = require_cosine_threshold(COSINE_SIMILARITY_THRESHOLD)
        vector = self._embed_texts([cleaned])[0]
        nearest = self._nearest_memory(
            vector,
            namespace=namespace,
            metadata_filter=metadata_filter,
        )
        similarity = None
        matched_id = None
        stored_text = None
        stored_metadata: dict[str, object] = {}
        if nearest is not None:
            match, similarity = nearest
            matched_id = match.id
            stored_metadata = dict(match.metadata)
            stored_text = stored_memory_text(stored_metadata, field_name)
        action = decide_memory_action(
            similarity,
            cleaned,
            stored_text,
            threshold,
        )
        if action == MEMORY_ACTION_SKIP and matched_id is not None:
            return MemoryWriteResult(
                action=action,
                memory_id=matched_id,
                similarity=similarity,
                matched_id=matched_id,
            )
        if action == MEMORY_ACTION_UPDATE and matched_id is not None:
            merged = merge_memory_metadata(
                stored_metadata,
                metadata,
                field_name,
                cleaned,
                store_text,
            )
            self.upsert_vector(
                matched_id,
                vector,
                merged,
                namespace=namespace,
            )
            return MemoryWriteResult(
                action=action,
                memory_id=matched_id,
                similarity=similarity,
                matched_id=matched_id,
            )
        if memory_id is None:
            slot_id = new_memory_id()
        else:
            slot_id = require_identifier(memory_id, 'memory_id')
        fresh = merge_memory_metadata(
            {},
            metadata,
            field_name,
            cleaned,
            store_text,
        )
        self.upsert_vector(slot_id, vector, fresh, namespace=namespace)
        return MemoryWriteResult(
            action=MEMORY_ACTION_REMEMBER,
            memory_id=slot_id,
            similarity=similarity,
            matched_id=matched_id,
        )

    def upsert_text_records(
        self,
        records: Sequence[Mapping[str, object]],
        namespace: str | None = None,
    ) -> int:
        """Записывает тексты в индекс со встроенной моделью Pinecone.

        Сервер сам строит эмбеддинг по полю, заданному при создании индекса.
        Запись применяется асинхронно и может появиться в поиске с задержкой.

        Args:
            records: Записи с ``_id`` или ``id`` и текстовыми полями.
            namespace: Пространство. Пустое имя становится ``__default__``.

        Returns:
            Число отправленных записей.
        """
        require_non_empty(list(records), 'records')
        prepared = [prepare_text_record(record) for record in records]
        ensure_unique_ids([str(record['_id']) for record in prepared])
        namespace_name = self._required_namespace(namespace)

        def action() -> object:
            return self._index.upsert_records(
                records=prepared,
                namespace=namespace_name,
            )

        response = self._call('upsert_text_records', action)
        return accepted_count(response, len(prepared))

    def upsert_schema_documents(
        self,
        documents: Sequence[Mapping[str, object]],
        namespace: str | None = None,
        batch_size: int = SCHEMA_BATCH_SIZE,
    ) -> int:
        """Записывает JSON-документы в документный индекс.

        Схема индекса решает, какие поля являются векторами, текстом
        со встроенным эмбеддингом или обычными метаданными.

        Args:
            documents: Документы с ``_id`` или ``id`` и полями схемы.
            namespace: Пространство. Пустое имя становится ``__default__``.
            batch_size: Размер пакета для длинного списка.

        Returns:
            Число принятых документов.
        """
        require_non_empty(list(documents), 'documents')
        prepared = [prepare_record_id(document) for document in documents]
        ensure_unique_ids([str(document['_id']) for document in prepared])
        namespace_name = self._required_namespace(namespace)
        size = require_positive_int(batch_size, 'batch_size')
        upsert = self._documents_method('upsert')
        batch_upsert = self._documents_method('batch_upsert')

        def action() -> object:
            if len(prepared) <= size:
                return upsert(namespace=namespace_name, documents=prepared)
            return batch_upsert(
                namespace=namespace_name,
                documents=prepared,
                batch_size=size,
                show_progress=False,
            )

        response = self._call('upsert_schema_documents', action)
        return self._require_full_write(
            response,
            len(prepared),
            'upsert_schema_documents',
        )

    def query_by_vector(
        self,
        values: Sequence[float],
        namespace: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        metadata_filter: Mapping[str, object] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
        sparse_vector: Mapping[str, object] | None = None,
    ) -> list[VectorMatch]:
        """Ищет ближайшие записи по готовому вектору запроса.

        Args:
            values: Плотный вектор запроса.
            namespace: Пространство. ``None`` — пространство по умолчанию.
            top_k: Сколько совпадений вернуть.
            metadata_filter: Фильтр метаданных Pinecone.
            include_metadata: Вернуть метаданные совпадений.
            include_values: Вернуть сами векторы совпадений.
            sparse_vector: Разреженная часть гибридного запроса.

        Returns:
            Совпадения от наиболее близкого к наименее близкому.
        """
        vector = normalize_vector(list(values))
        limit = require_positive_int(top_k, 'top_k')
        namespace_name = self._namespace(namespace)
        metadata_filter_copy = copy_filter(metadata_filter)
        sparse_copy = normalize_sparse_vector(sparse_vector)

        def action() -> object:
            return self._index.query(
                top_k=limit,
                vector=vector,
                namespace=namespace_name,
                filter=metadata_filter_copy,
                include_values=include_values,
                include_metadata=include_metadata,
                sparse_vector=sparse_copy,
            )

        response = self._call('query_by_vector', action)
        return normalize_vector_matches(response)

    def query_by_vector_id(
        self,
        vector_id: str,
        namespace: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        metadata_filter: Mapping[str, object] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[VectorMatch]:
        """Ищет соседей уже сохранённого вектора по его идентификатору.

        Args:
            vector_id: Идентификатор вектора, который станет запросом.
            namespace: Пространство. ``None`` — пространство по умолчанию.
            top_k: Сколько совпадений вернуть.
            metadata_filter: Фильтр метаданных Pinecone.
            include_metadata: Вернуть метаданные совпадений.
            include_values: Вернуть сами векторы совпадений.

        Returns:
            Совпадения, ближайшие к сохранённому вектору.
        """
        cleaned_id = require_identifier(vector_id, 'id')
        limit = require_positive_int(top_k, 'top_k')
        namespace_name = self._namespace(namespace)
        metadata_filter_copy = copy_filter(metadata_filter)

        def action() -> object:
            return self._index.query(
                top_k=limit,
                id=cleaned_id,
                namespace=namespace_name,
                filter=metadata_filter_copy,
                include_values=include_values,
                include_metadata=include_metadata,
            )

        response = self._call('query_by_vector_id', action)
        return normalize_vector_matches(response)

    def query_by_text(
        self,
        text: str,
        namespace: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        metadata_filter: Mapping[str, object] | None = None,
        include_metadata: bool = True,
        include_values: bool = False,
    ) -> list[VectorMatch]:
        """Считает эмбеддинг текста через OpenAI и ищет по этому вектору.

        Подходит для обычного векторного индекса. Модель должна быть той же,
        которой записывались документы.

        Args:
            text: Текст запроса.
            namespace: Пространство. ``None`` — пространство по умолчанию.
            top_k: Сколько совпадений вернуть.
            metadata_filter: Фильтр метаданных Pinecone.
            include_metadata: Вернуть метаданные, включая исходный текст.
            include_values: Вернуть векторы совпадений.

        Returns:
            Совпадения векторного поиска.
        """
        cleaned = require_text(text, 'text')
        vector = self._embed_texts([cleaned])[0]
        return self.query_by_vector(
            vector,
            namespace=namespace,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_metadata=include_metadata,
            include_values=include_values,
        )

    def search_by_text(
        self,
        text: str,
        namespace: str | None = None,
        top_k: int = DEFAULT_TOP_K,
        metadata_filter: Mapping[str, object] | None = None,
        fields: Sequence[str] | None = None,
        text_input_key: str = DEFAULT_TEXT_FIELD,
    ) -> list[TextMatch]:
        """Ищет по тексту в индексе со встроенной моделью Pinecone.

        Эмбеддинг запроса считает сервер. Ключ в ``inputs`` должен совпадать
        с входным полем модели индекса.

        Args:
            text: Текст запроса.
            namespace: Пространство. Пустое имя становится ``__default__``.
            top_k: Сколько совпадений вернуть.
            metadata_filter: Фильтр метаданных.
            fields: Какие поля вернуть. ``None`` — все сохранённые поля.
            text_input_key: Имя текстового входа модели, обычно ``text``.

        Returns:
            Совпадения с полями записей.
        """
        cleaned = require_text(text, 'text')
        input_key = require_text(text_input_key, 'text_input_key')
        limit = require_positive_int(top_k, 'top_k')
        namespace_name = self._required_namespace(namespace)
        metadata_filter_copy = copy_filter(metadata_filter)
        selected_fields = None
        if fields is not None:
            selected_fields = [
                require_text(name, 'fields') for name in fields
            ]
        search = self._index_method('search')

        def action() -> object:
            return search(
                namespace=namespace_name,
                top_k=limit,
                inputs={input_key: cleaned},
                filter=metadata_filter_copy,
                fields=selected_fields,
            )

        response = self._call('search_by_text', action)
        return normalize_text_matches(response)

    def search_documents_by_vector(
        self,
        values: Sequence[float],
        namespace: str | None = None,
        field: str = DEFAULT_VECTOR_FIELD,
        top_k: int = DEFAULT_TOP_K,
        include_fields: Sequence[str] | None = None,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[StoredDocument]:
        """Ищет документы документного индекса по плотному вектору.

        Args:
            values: Вектор запроса.
            namespace: Пространство. Пустое имя становится ``__default__``.
            field: Имя векторного поля в схеме индекса.
            top_k: Сколько документов вернуть.
            include_fields: Поля ответа. ``None`` возвращает все поля.
            metadata_filter: Фильтр метаданных.

        Returns:
            Документы с оценкой близости.
        """
        vector = normalize_vector(list(values))
        vector_field = require_text(field, 'field')
        limit = require_positive_int(top_k, 'top_k')
        namespace_name = self._required_namespace(namespace)
        selected_fields = resolve_include_fields(include_fields)
        metadata_filter_copy = copy_filter(metadata_filter)
        search = self._documents_method('search')
        score = DenseVectorQuery(values=vector, field=vector_field)

        def action() -> object:
            return search(
                namespace=namespace_name,
                top_k=limit,
                score_by=[score],
                include_fields=selected_fields,
                filter=metadata_filter_copy,
            )

        response = self._call('search_documents_by_vector', action)
        return normalize_stored_documents(response)

    def search_documents_by_text(
        self,
        text: str,
        namespace: str | None = None,
        field: str = DEFAULT_TEXT_FIELD,
        top_k: int = DEFAULT_TOP_K,
        score_type: str = 'embed',
        include_fields: Sequence[str] | None = None,
        metadata_filter: Mapping[str, object] | None = None,
    ) -> list[StoredDocument]:
        """Ищет документы по тексту.

        ``embed`` и ``sparse_embed`` используют встроенную модель поля.
        ``text`` — полнотекстовый поиск по полю схемы.

        Args:
            text: Текст запроса.
            namespace: Пространство. Пустое имя становится ``__default__``.
            field: Поле схемы, по которому считается оценка.
            top_k: Сколько документов вернуть.
            score_type: ``embed``, ``sparse_embed`` или ``text``.
            include_fields: Поля ответа. ``None`` возвращает все поля.
            metadata_filter: Фильтр метаданных.

        Returns:
            Документы с оценкой.
        """
        cleaned = require_text(text, 'text')
        text_field = require_text(field, 'field')
        limit = require_positive_int(top_k, 'top_k')
        namespace_name = self._required_namespace(namespace)
        selected_fields = resolve_include_fields(include_fields)
        metadata_filter_copy = copy_filter(metadata_filter)
        score = build_text_score(score_type, cleaned, text_field)
        search = self._documents_method('search')

        def action() -> object:
            return search(
                namespace=namespace_name,
                top_k=limit,
                score_by=[score],
                include_fields=selected_fields,
                filter=metadata_filter_copy,
            )

        response = self._call('search_documents_by_text', action)
        return normalize_stored_documents(response)

    def fetch_vectors(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
    ) -> list[StoredVector]:
        """Читает векторы по идентификаторам, без ранжирования.

        Args:
            ids: Идентификаторы записей.
            namespace: Пространство. ``None`` — пространство по умолчанию.

        Returns:
            Найденные векторы в порядке запроса. Неизвестные id пропускаются.
        """
        require_non_empty(list(ids), 'ids')
        cleaned_ids = [require_identifier(vector_id, 'id') for vector_id in ids]
        namespace_name = self._namespace(namespace)

        def action() -> object:
            return self._index.fetch(ids=cleaned_ids, namespace=namespace_name)

        response = self._call('fetch_vectors', action)
        return normalize_stored_vectors(response, cleaned_ids)

    def fetch_documents(
        self,
        ids: Sequence[str],
        namespace: str | None = None,
        include_fields: Sequence[str] | None = None,
    ) -> list[StoredDocument]:
        """Читает документы документного индекса по идентификаторам.

        Args:
            ids: Идентификаторы документов.
            namespace: Пространство. Пустое имя становится ``__default__``.
            include_fields: Поля ответа. ``None`` возвращает все поля.

        Returns:
            Найденные документы. Неизвестные id пропускаются.
        """
        require_non_empty(list(ids), 'ids')
        cleaned_ids = [require_identifier(document_id, 'id') for document_id in ids]
        namespace_name = self._required_namespace(namespace)
        selected_fields = resolve_include_fields(include_fields)
        fetch = self._documents_method('fetch')

        def action() -> object:
            return fetch(
                namespace=namespace_name,
                ids=cleaned_ids,
                include_fields=selected_fields,
            )

        response = self._call('fetch_documents', action)
        return normalize_stored_documents(response)

    def delete_vectors(
        self,
        ids: Sequence[str] | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        delete_all: bool = False,
        namespace: str | None = None,
    ) -> None:
        """Удаляет векторы по id, по фильтру или очищает пространство.

        Нужен ровно один способ выбора: ``ids``, ``metadata_filter``
        или ``delete_all=True``.

        Args:
            ids: Идентификаторы для удаления.
            metadata_filter: Фильтр записей, которые нужно удалить.
            delete_all: Удалить все векторы пространства.
            namespace: Пространство. ``None`` — пространство по умолчанию.
        """
        cleaned_ids, metadata_filter_copy = self._delete_selector(
            ids,
            metadata_filter,
            delete_all,
        )
        namespace_name = self._namespace(namespace)

        def action() -> object:
            if cleaned_ids is not None:
                self._index.delete(ids=cleaned_ids, namespace=namespace_name)
                return None
            if metadata_filter_copy is not None:
                self._index.delete(
                    filter=metadata_filter_copy,
                    namespace=namespace_name,
                )
                return None
            self._index.delete(delete_all=True, namespace=namespace_name)
            return None

        self._call('delete_vectors', action)

    def delete_documents(
        self,
        ids: Sequence[str] | None = None,
        metadata_filter: Mapping[str, object] | None = None,
        delete_all: bool = False,
        namespace: str | None = None,
    ) -> None:
        """Удаляет документы по id, по фильтру или очищает пространство.

        Args:
            ids: Идентификаторы документов.
            metadata_filter: Фильтр документов.
            delete_all: Удалить все документы пространства.
            namespace: Пространство. Пустое имя становится ``__default__``.
        """
        cleaned_ids, metadata_filter_copy = self._delete_selector(
            ids,
            metadata_filter,
            delete_all,
        )
        namespace_name = self._required_namespace(namespace)
        delete = self._documents_method('delete')

        def action() -> object:
            if cleaned_ids is not None:
                return delete(namespace=namespace_name, ids=cleaned_ids)
            if metadata_filter_copy is not None:
                return delete(
                    namespace=namespace_name,
                    filter=metadata_filter_copy,
                )
            return delete(namespace=namespace_name, delete_all=True)

        self._call('delete_documents', action)

    def describe_stats(self) -> IndexStats:
        """Возвращает размерность индекса и число векторов по пространствам.

        Счётчики отстают от записи: только что отправленный вектор может
        ещё не попасть в статистику.

        Returns:
            Сводка индекса.
        """

        def action() -> object:
            return self._index.describe_index_stats()

        stats = self._call('describe_stats', action)
        raw_namespaces = read_attr(stats, 'namespaces', {})
        namespaces: dict[str, int] = {}
        if isinstance(raw_namespaces, Mapping):
            for name, summary in raw_namespaces.items():
                namespaces[str(name)] = as_int(
                    read_attr(summary, 'vector_count', 0),
                )
        dimension = read_attr(stats, 'dimension', None)
        resolved_dimension = None
        if dimension is not None:
            resolved_dimension = as_int(dimension)
        return IndexStats(
            dimension=resolved_dimension,
            total_vector_count=as_int(read_attr(stats, 'total_vector_count', 0)),
            namespaces=namespaces,
        )

    def close(self) -> None:
        """Закрывает клиент индекса. Повторный вызов ничего не делает."""
        if self._closed:
            return
        self._closed = True
        close_index = getattr(self._index, 'close', None)
        if callable(close_index):
            close_index()

    def __enter__(self) -> PineconeManager:
        """Возвращает сам менеджер для блока ``with``."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: object,
    ) -> None:
        """Закрывает соединение при выходе из блока ``with``."""
        self.close()

    def _open_index(self, api_key: str, index_name: str) -> object:
        """Создаёт клиент данных индекса.

        Args:
            api_key: Ключ Pinecone.
            index_name: Имя индекса.

        Returns:
            Клиент ``Index``.

        Raises:
            PineconeManagerError: Если подключение не удалось.
        """
        try:
            client = Pinecone(api_key=api_key)
            index_factory = getattr(client, 'index', None)
            if not callable(index_factory):
                raise PineconeManagerError(
                    code='unsupported_operation',
                    message='Установленный клиент Pinecone не открывает индекс по имени.',
                    details={'operation': 'index'},
                )
            # Соединение с плоскостью данных резолвит хост индекса.
            return index_factory(name=index_name)
        except PineconeManagerError:
            raise
        except Exception as error:
            logger.error(
                'Не удалось открыть индекс Pinecone: error_type=%s',
                type(error).__name__,
            )
            raise PineconeManagerError(
                code='config_error',
                message='Не удалось подключиться к индексу Pinecone.',
                details={
                    'index_name': index_name,
                    'error_type': type(error).__name__,
                },
            ) from error

    def _nearest_memory(
        self,
        vector: Sequence[float],
        namespace: str | None,
        metadata_filter: Mapping[str, object] | None,
    ) -> tuple[VectorMatch, float] | None:
        """Находит ближайший сохранённый фрагмент и считает косинус.

        Если поиск не вернул компоненты вектора, они дочитываются по id.

        Args:
            vector: Эмбеддинг текущего сообщения.
            namespace: Пространство поиска.
            metadata_filter: Фильтр записей, среди которых искать.

        Returns:
            Ближайшее совпадение и сходство. ``None``, если память пуста
            или векторы нельзя сравнить.
        """
        matches = self.query_by_vector(
            vector,
            namespace=namespace,
            top_k=MEMORY_NEIGHBOR_COUNT,
            metadata_filter=metadata_filter,
            include_metadata=True,
            include_values=True,
        )
        if len(matches) == 0:
            return None
        missing_ids = [
            match.id
            for match in matches
            if match.id != '' and len(match.values) == 0
        ]
        fetched: dict[str, list[float]] = {}
        if len(missing_ids) > 0:
            for stored in self.fetch_vectors(missing_ids, namespace=namespace):
                fetched[stored.id] = stored.values
        comparable: list[VectorMatch] = []
        for match in matches:
            values = match.values
            if len(values) == 0:
                values = fetched.get(match.id, [])
            if len(values) != len(vector):
                continue
            comparable.append(
                VectorMatch(
                    id=match.id,
                    score=match.score,
                    metadata=match.metadata,
                    values=values,
                )
            )
        if len(matches) > 0 and len(comparable) == 0:
            logger.warning(
                'Сохранённые фрагменты нашлись, но их векторы нельзя сравнить.',
            )
        return closest_memory(vector, comparable)

    def _namespace(self, namespace: str | None) -> str:
        """Выбирает пространство векторных операций.

        Args:
            namespace: Явно переданное имя или ``None``.

        Returns:
            Имя пространства. Пустая строка означает пространство по умолчанию.
        """
        if namespace is None:
            return self._default_namespace
        return namespace

    def _required_namespace(self, namespace: str | None) -> str:
        """Выбирает непустое пространство для Records API и Documents API.

        Args:
            namespace: Явно переданное имя или ``None``.

        Returns:
            Имя пространства. Пустое заменяется на ``__default__``.
        """
        resolved = self._namespace(namespace)
        if resolved.strip() == '':
            return INTEGRATED_DEFAULT_NAMESPACE
        return resolved

    def _openai_client(self) -> OpenAI:
        """Создаёт клиент OpenAI при первом текстовом запросе.

        Returns:
            Клиент эмбеддингов.

        Raises:
            PineconeManagerError: Если ключ OpenAI не задан.
        """
        if self._openai is not None:
            return self._openai
        api_key = require_config(self._openai_api_key, OPENAI_API_KEY_ENV)
        if self._openai_base_url is None:
            self._openai = OpenAI(api_key=api_key)
            return self._openai
        self._openai = OpenAI(api_key=api_key, base_url=self._openai_base_url)
        return self._openai

    def _embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Считает эмбеддинги пакетами и сохраняет порядок текстов.

        Args:
            texts: Непустой список текстов.

        Returns:
            Векторы в том же порядке, что и тексты.
        """
        client = self._openai_client()
        embeddings: list[list[float]] = []
        for batch in split_batches(list(texts), EMBEDDING_BATCH_SIZE):
            ordered = self._embed_batch(client, batch)
            embeddings.extend(ordered)
        return embeddings

    def _embed_batch(
        self,
        client: OpenAI,
        batch: Sequence[str],
    ) -> list[list[float]]:
        """Запрашивает эмбеддинги одного пакета.

        Args:
            client: Клиент OpenAI.
            batch: Тексты пакета.

        Returns:
            Векторы в порядке пакета.

        Raises:
            PineconeManagerError: Если сервис эмбеддингов вернул ошибку
                или неполный ответ.
        """
        try:
            if self._embedding_dimensions is None:
                response = client.embeddings.create(
                    model=self._embedding_model,
                    input=list(batch),
                )
            else:
                response = client.embeddings.create(
                    model=self._embedding_model,
                    input=list(batch),
                    dimensions=self._embedding_dimensions,
                )
        except Exception as error:
            logger.error(
                'Ошибка эмбеддинга: error_type=%s',
                type(error).__name__,
            )
            raise PineconeManagerError(
                code='embedding_error',
                message='Не удалось посчитать эмбеддинг текста.',
                details={'error_type': type(error).__name__},
            ) from error
        ordered = sorted(response.data, key=lambda item: item.index)
        if len(ordered) != len(batch):
            raise PineconeManagerError(
                code='embedding_error',
                message='Сервис эмбеддингов вернул неполный набор векторов.',
                details={'expected': len(batch), 'received': len(ordered)},
            )
        return [list(item.embedding) for item in ordered]

    def _index_method(self, name: str) -> Callable[..., object]:
        """Возвращает метод клиента индекса или объясняет, почему его нет.

        Args:
            name: Имя метода.

        Returns:
            Вызываемый метод индекса.
        """
        method = getattr(self._index, name, None)
        if not callable(method):
            raise PineconeManagerError(
                code='unsupported_operation',
                message=f'Операция индекса «{name}» недоступна в этом клиенте.',
                details={'operation': name},
            )
        return method

    def _documents_method(self, name: str) -> Callable[..., object]:
        """Возвращает метод Documents API.

        Args:
            name: Имя операции документов.

        Returns:
            Вызываемый метод.
        """
        documents = getattr(self._index, 'documents', None)
        method = None if documents is None else getattr(documents, name, None)
        if not callable(method):
            raise PineconeManagerError(
                code='unsupported_operation',
                message=f'Операция документов «{name}» недоступна.',
                details={'operation': name},
            )
        return method

    def _delete_selector(
        self,
        ids: Sequence[str] | None,
        metadata_filter: Mapping[str, object] | None,
        delete_all: bool,
    ) -> tuple[list[str] | None, dict[str, object] | None]:
        """Проверяет, что удаление задано ровно одним способом.

        Args:
            ids: Идентификаторы или ``None``.
            metadata_filter: Фильтр или ``None``.
            delete_all: Признак полной очистки.

        Returns:
            Пара из списка id и копии фильтра. Заполнена ровно одна часть.
        """
        selected = 0
        if ids is not None:
            selected += 1
        if metadata_filter is not None:
            selected += 1
        if delete_all:
            selected += 1
        if selected != 1:
            raise PineconeManagerError(
                code='validation_error',
                message=(
                    'Удаление требует ровно один способ: '
                    'ids, metadata_filter или delete_all.'
                ),
                details={'selectors': selected},
            )
        if ids is None:
            return None, copy_filter(metadata_filter)
        require_non_empty(list(ids), 'ids')
        return [require_identifier(item, 'id') for item in ids], None

    def _require_full_write(
        self,
        response: object,
        expected: int,
        operation: str,
    ) -> int:
        """Возвращает счётчик записи или поднимает ошибку неполного пакета.

        Args:
            response: Ответ upsert.
            expected: Сколько записей отправляли.
            operation: Имя операции для подробностей.

        Returns:
            Число принятых записей.
        """
        failed = failed_item_count(response)
        if failed > 0:
            raise PineconeManagerError(
                code='pinecone_error',
                message='Часть записей не была сохранена.',
                details={
                    'operation': operation,
                    'failed_item_count': failed,
                    'errors': error_messages(response),
                },
            )
        return accepted_count(response, expected)

    def _call(self, operation: str, action: Callable[[], TResult]) -> TResult:
        """Выполняет запрос к Pinecone и переводит сбой в ошибку модуля.

        Args:
            operation: Имя операции для журнала и подробностей.
            action: Вызов без аргументов.

        Returns:
            Результат ``action``.

        Raises:
            PineconeManagerError: Если клиент закрыт или запрос не выполнен.
        """
        if self._closed:
            raise PineconeManagerError(
                code='client_closed',
                message='Соединение с Pinecone уже закрыто.',
                details={'operation': operation},
            )
        try:
            return action()
        except PineconeManagerError:
            raise
        except Exception as error:
            logger.error(
                'Операция Pinecone завершилась ошибкой: operation=%s error_type=%s',
                operation,
                type(error).__name__,
            )
            raise PineconeManagerError(
                code='pinecone_error',
                message=f'Не удалось выполнить операцию «{operation}».',
                details={
                    'operation': operation,
                    'error_type': type(error).__name__,
                },
            ) from error
