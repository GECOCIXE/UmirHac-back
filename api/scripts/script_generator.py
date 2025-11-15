from openai import OpenAI
import os
import json
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Literal, Dict, Any, Optional, Union
from dotenv import load_dotenv

load_dotenv()

# Настройка клиента
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

# Базовая "строгая" модель
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")  # => additionalProperties: false

# 1. Модель для первого этапа - ПЛАНИРОВАНИЕ
class ScriptPlan(StrictModel):
    total_blocks: int = Field(..., ge=8, le=15, description="Общее количество блоков в сценарии")
    block_sequence: List[Literal["scene_heading", "action", "character", "dialogue", "transition"]] = Field(
        ...,
        description="Последовательность типов блоков для сценария"
    )
    story_summary: str = Field(..., description="Краткое описание сюжета сценария")

# 2. Модели блоков для второго этапа
class SceneHeading(StrictModel):
    location_type: Literal["INT", "EXT", "INT/EXT"]
    location: str
    time: Literal["DAY", "NIGHT", "MORNING", "EVENING"]

class Action(StrictModel):
    description: str

class Character(StrictModel):
    name: str
    parenthetical: Optional[str] = None

class Dialogue(StrictModel):
    text: str

class Transition(StrictModel):
    transition_type: Literal["CUT TO", "FADE TO", "DISSOLVE TO"]

# 3. Модель для конкретного блока
class ScriptBlock(StrictModel):
    block_type: Literal["scene_heading", "action", "character", "dialogue", "transition"]
    content: Union[SceneHeading, Action, Character, Dialogue, Transition]

# 4. Модель для финального сценария
class FinalScript(StrictModel):
    blocks: List[ScriptBlock]

# 5. Функция для первого этапа - ПЛАНИРОВАНИЕ
def create_script_plan(product_description: str) -> ScriptPlan:
    """Создает план сценария с последовательностью блоков"""
    print(f"ЗАДАНЫЙ ПРОМПТ: {product_description}")
    completion = client.beta.chat.completions.parse(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты - профессиональный сценарист. Создай подробный план для рекламного сценария. "
                    "Пиши ТОЛЬКО на русском языке."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Описание продукта для рекламы:\n{product_description}\n\n"
                    "Создай план сценария, включающий:\n"
                    "1. Общее количество блоков (от 8 до 15)\n"
                    "2. Последовательность типов блоков (могут быть только: scene_heading, action, character, dialogue, transition)\n"
                    "3. Краткое описание сюжета\n\n"
                    "ВАЖНО: Не добавляй никаких дополнительных полей, только запрашиваемые."
                )
            }
        ],
        response_format=ScriptPlan,
    )

    return completion.choices[0].message.parsed

# 6. Функция для второго этапа - ГЕНЕРАЦИЯ
def generate_script_blocks(product_description: str, script_plan: ScriptPlan) -> FinalScript:
    """Генерирует конкретные блоки сценария на основе плана"""

    # Формируем подробную инструкцию с последовательностью блоков
    block_sequence_str = "\n".join([
        f"{i+1}. {block_type}"
        for i, block_type in enumerate(script_plan.block_sequence)
    ])

    completion = client.beta.chat.completions.parse(
        model="openai/gpt-4.1-nano",
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты - профессиональный сценарист. Создай конкретные блоки сценария "
                    "в соответствии с предоставленным планом. Каждый блок должен быть логичным "
                    "и соответствовать стандартам сценарного формата. "
                    "Пиши ТОЛЬКО на русском языке."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Описание продукта:\n{product_description}\n\n"
                    f"План сценария:\n"
                    f"Общее количество блоков: {script_plan.total_blocks}\n"
                    f"Сюжет: {script_plan.story_summary}\n\n"
                    f"Последовательность блоков:\n{block_sequence_str}\n\n"
                    "Создай JSON сценарий с точным количеством блоков в указанной последовательности. "
                    "Для каждого блока укажи:\n"
                    "- block_type: тип блока (могут быть только: scene_heading, action, character, dialogue, transition)\n"
                    "- content: содержимое блока в соответствии с его типом\n\n"
                    "ВАЖНО: Строго следуй указанной последовательности и количеству блоков."
                )
            }
        ],
        response_format=FinalScript,
    )

    return completion.choices[0].message.parsed

# 7. Пост-обработка: применение форматирования и удаление дубликатов
def post_process_script(product_description: str, blocks: List[Dict], output_file: str = "final_script.json"):
    """Применяет форматирование, удаляет дубликаты и сохраняет в JSON с индексацией блоков"""

    # Стандартные параметры форматирования
    STANDARD_FORMATTING = {
        "scene_heading": {
            "alignment": "left",
            "font_case": "uppercase",
            "indent": 0.0,
            "font_size": 12,
            "font_family": "Courier New"
        },
        "action": {
            "alignment": "left",
            "font_case": "sentence",
            "indent": 0.0,
            "max_lines": 4,
            "font_size": 12,
            "font_family": "Courier New"
        },
        "character": {
            "alignment": "center",
            "font_case": "uppercase",
            "indent": 3.7,
            "font_size": 12,
            "font_family": "Courier New"
        },
        "dialogue": {
            "alignment": "center",
            "font_case": "sentence",
            "indent": 2.3,
            "width": 2.5,
            "font_size": 12,
            "font_family": "Courier New"
        },
        "transition": {
            "alignment": "right",
            "font_case": "uppercase",
            "indent": 5.5,
            "font_size": 12,
            "font_family": "Courier New"
        }
    }

    # Удаляем дубликаты (последовательные одинаковые блоки)
    processed_blocks = []

    for i, block in enumerate(blocks):
        block_type = block["block_type"]
        content = block["content"]

        # Пропускаем блок, если он идентичен предыдущему
        if i > 0 and processed_blocks:
            prev_block = processed_blocks[-1]
            if (prev_block["type"] == block_type and
                prev_block["content"] == content):
                print(f"⚠️ Пропущен дубликат блока типа {block_type}")
                continue

        # Применяем форматирование
        formatting = STANDARD_FORMATTING.get(block_type, {})

        # Формируем финальный блок
        final_block = {
            "type": block_type,
            "content": content,
            "formatting": formatting
        }

        processed_blocks.append(final_block)

    # Добавляем индексы всем блокам
    for idx, block in enumerate(processed_blocks, 1):
        block["index"] = idx

    # Сохраняем в JSON
    script_data = {
        "product_description": product_description,
        "original_blocks_count": len(blocks),
        "final_blocks_count": len(processed_blocks),
        "blocks": processed_blocks
    }

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(script_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Сценарий успешно сохранен в {output_file}")
    print(f"Статистика: {len(blocks)} исходных блоков → {len(processed_blocks)} финальных блоков")

    return processed_blocks

# 8. Основная функция
def generate_ad_script(product_description: str, output_file: str = "final_script.json"):
    """Основная функция для генерации рекламного сценария"""

    print("🎬 НАЧАЛО ГЕНЕРАЦИИ СЦЕНАРИЯ")
    print("=" * 50)

    # Этап 1: Планирование
    print("\n📈 ЭТАП 1: ПЛАНИРОВАНИЕ СЦЕНАРИЯ")
    print("-" * 40)

    try:
        script_plan = create_script_plan(product_description)
        print(f"ВЕСЬ ПЛАН:\n{script_plan}")
        print(f"✅ План создан успешно!")
        print(f"📊 Количество блоков: {script_plan.total_blocks}")
        print(f"📖 Сюжет: {script_plan.story_summary}")
        print(f"🔄 Последовательность: {script_plan.block_sequence}")
    except Exception as e:
        print(f"❌ Ошибка на этапе планирования: {e}")
        raise e

    # Этап 2: Генерация блоков
    print("\n📝 ЭТАП 2: ГЕНЕРАЦИЯ БЛОКОВ СЦЕНАРИЯ")
    print("-" * 40)

    try:
        final_script = generate_script_blocks(product_description, script_plan)
        print(f"ВСЕ БЛОКИ:\n{final_script}")
        print(f"✅ Блоки сгенерированы успешно!")
        print(f"🧱 Сгенерировано блоков: {len(final_script.blocks)}")
    except Exception as e:
        print(f"❌ Ошибка на этапе генерации: {e}")
        raise e

    # Преобразуем блоки в словари для пост-обработки
    blocks_dict = [block.model_dump() for block in final_script.blocks]

    # Пост-обработка
    print("\n✨ ЭТАП 3: ПОСТ-ОБРАБОТКА")
    print("-" * 40)

    try:
        processed_blocks = post_process_script(product_description, blocks_dict, output_file)
        return processed_blocks
    except Exception as e:
        print(f"❌ Ошибка на этапе пост-обработки: {e}")
        raise e

# 9. Пример использования
if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("❌ Ошибка: Не установлен OPENROUTER_API_KEY")
        print("Установите его командой: set OPENROUTER_API_KEY=ваш_ключ")
        exit(1)

    # Описание продукта для рекламы
    product_description = (
        "Шампунь 'Джумайсынба' - премиальный уход для волос с натуральными ингредиентами. "
        "Дарит волосам невероятный блеск, мягкость и объем. Подходит для всех типов волос. "
        "Создан с использованием экстрактов редких растений и витаминного комплекса."
    )

    # Генерация сценария
    result = generate_ad_script(
        product_description=product_description,
        output_file="jumaisynba_script.json"
    )

    if result:
        print("\n" + "="*60)
        print("🎉 ГЕНЕРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
        print("="*60)

        # Выводим первые 3 блока для демонстрации
        print("\n🔍 ПРИМЕР ПЕРВЫХ БЛОКОВ:")
        for i, block in enumerate(result[:3], 1):
            print(f"\n{i}. {block['type'].upper()}")
            print(f"   Содержимое: {json.dumps(block['content'], ensure_ascii=False, indent=2)}")
            print(f"   Форматирование: {json.dumps(block['formatting'], indent=2)}")