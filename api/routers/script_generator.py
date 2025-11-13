from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import os
import json
import base64
from pathlib import Path
from datetime import datetime
import requests
from PIL import Image
import io

# Импорт из других модулей
from ..schemas.schemas import get_db, User, Project, ProjectStatus, ScenarioElementImage
from ..scripts.script_generator import generate_ad_script

router = APIRouter(
    prefix="/script-generator",
    tags=["script-generator"]
)

# Модели Pydantic для запросов и ответов
class GenerateScriptRequest(BaseModel):
    user_id: int
    product_description: str

class GenerateImageRequest(BaseModel):
    user_id: int
    image_description: str

class GenerateScriptResponse(BaseModel):
    project_id: int
    status: str
    message: str

class GenerateImageResponse(BaseModel):
    project_id: int
    status: str
    message: str

class GenerateElementImageRequest(BaseModel):
    project_id: int
    element_index: Optional[int] = None  # Если None, то генерировать для всех элементов

class EditElementImageRequest(BaseModel):
    project_id: int
    element_index: int
    image_description: str

class ProjectStatusResponse(BaseModel):
    project_id: int
    user_id: int
    status: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    result_path: Optional[str] = None
    image_path: Optional[str] = None
    image_description: Optional[str] = None

@router.post("/generate", response_model=GenerateScriptResponse)
async def generate_script_endpoint(
    request: GenerateScriptRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Эндпоинт для генерации рекламного сценария
    Принимает ID пользователя и описание продукта
    """
    # Проверяем, что пользователь существует
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Создаем запись проекта в БД со статусом "in_progress"
    project = Project(
        user_id=request.user_id,
        status=ProjectStatus.in_progress
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    # Определяем путь для сохранения JSON файла
    user_data_dir = Path("api/users_data") / str(request.user_id) / str(project.id)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    output_file = user_data_dir / f"{request.user_id}_{project.id}_scenario.json"

    # Добавляем задачу в фон для генерации сценария
    background_tasks.add_task(
        process_script_generation,
        project.id,
        request.product_description,
        str(output_file),
        db
    )

    return GenerateScriptResponse(
        project_id=project.id,
        status="in_progress",
        message=f"Script generation started for project {project.id}"
    )

def process_script_generation(
    project_id: int,
    product_description: str,
    output_file_path: str,
    db: Session
):
    """
    Фоновая задача для генерации сценария
    """
    try:
        # Вызываем функцию генерации сценария
        result = generate_ad_script(
            product_description=product_description,
            output_file=output_file_path
        )

        if result:
            # Обновляем статус проекта на "completed" и сохраняем путь к файлу
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = ProjectStatus.completed
                project.result_path = output_file_path
                db.commit()
        else:
            # Если произошла ошибка при генерации, ставим статус "failed"
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                project.status = ProjectStatus.failed
                db.commit()

    except Exception as e:
        # В случае ошибки обновляем статус на "failed"
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            db.commit()

@router.post("/generate_image", response_model=GenerateImageResponse)
async def generate_image_endpoint(
    request: GenerateImageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Эндпоинт для генерации изображения
    Принимает ID пользователя и описание изображения
    """
    # Проверяем, что пользователь существует
    user = db.query(User).filter(User.id == request.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Создаем запись проекта в БД со статусом "in_progress"
    project = Project(
        user_id=request.user_id,
        status=ProjectStatus.in_progress,
        image_description=request.image_description
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    # Определяем путь для сохранения PNG файла
    user_data_dir = Path("api/users_data") / str(request.user_id) / str(project.id)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    image_file = user_data_dir / f"{request.user_id}_{project.id}_image.png"

    # Добавляем задачу в фон для генерации изображения
    background_tasks.add_task(
        process_image_generation,
        project.id,
        request.image_description,
        str(image_file),
        db
    )

    return GenerateImageResponse(
        project_id=project.id,
        status="in_progress",
        message=f"Image generation started for project {project.id}"
    )

def process_image_generation(
    project_id: int,
    image_description: str,
    output_file_path: str,
    db: Session
):
    """
    Фоновая задача для генерации изображения
    """
    try:
        # Подключаемся к серверу заглушке для генерации изображения
        import httpx
        import asyncio

        # Асинхронная функция для генерации изображения
        async def generate_image_async():
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:3339/generate_image",
                    json={"prompt": image_description},
                    timeout=30.0  # таймаут 30 секунд
                )

                if response.status_code == 200:
                    result = response.json()
                    image_data = result["image"]

                    # Декодируем base64 изображение и сохраняем в файл
                    image_bytes = base64.b64decode(image_data)
                    with open(output_file_path, "wb") as f:
                        f.write(image_bytes)

                    # Обновляем статус проекта на "completed" и сохраняем путь к файлу
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project:
                        project.status = ProjectStatus.completed
                        project.image_path = output_file_path
                        db.commit()

                    return True
                else:
                    # Если произошла ошибка при генерации, ставим статус "failed"
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project:
                        project.status = ProjectStatus.failed
                        db.commit()
                    return False

        # Запускаем асинхронную функцию
        asyncio.run(generate_image_async())

    except Exception as e:
        # В случае ошибки обновляем статус на "failed"
        print(f"Error during image generation: {e}")
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            db.commit()


class EditImageRequest(BaseModel):
    project_id: int
    image_description: str


@router.post("/edit_image", response_model=GenerateImageResponse)
async def edit_image_endpoint(
    request: EditImageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Эндпоинт для редактирования изображения
    Принимает ID проекта и новое описание изображения
    """
    # Проверяем, что проект существует
    project = db.query(Project).filter(Project.id == request.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Проверяем, что у проекта есть изображение для редактирования
    if not project.image_path or not os.path.exists(project.image_path):
        raise HTTPException(status_code=404, detail="Original image not found")

    # Обновляем статус проекта на "in_progress" и сохраняем новое описание
    project.status = ProjectStatus.in_progress
    project.image_description = request.image_description
    db.commit()

    # Определяем путь для сохранения отредактированного PNG файла
    user_data_dir = Path("api/users_data") / str(project.user_id) / str(project.id)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    image_file = user_data_dir / f"{project.user_id}_{project.id}_image.png"

    # Добавляем задачу в фон для редактирования изображения
    background_tasks.add_task(
        process_image_editing,
        project.id,
        request.image_description,
        project.image_path,  # путь к оригинальному изображению
        str(image_file),
        db
    )

    return GenerateImageResponse(
        project_id=project.id,
        status="in_progress",
        message=f"Image editing started for project {project.id}"
    )


def process_image_editing(
    project_id: int,
    image_description: str,
    original_image_path: str,
    output_file_path: str,
    db: Session
):
    """
    Фоновая задача для редактирования изображения
    """
    try:
        import httpx
        import asyncio
        import base64

        # Асинхронная функция для редактирования изображения
        async def edit_image_async():
            # Читаем оригинальное изображение и кодируем в base64
            with open(original_image_path, "rb") as f:
                original_image_bytes = f.read()
            original_image_base64 = base64.b64encode(original_image_bytes).decode('utf-8')

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:3339/edit_image",
                    json={
                        "prompt": image_description,
                        "image_base64": original_image_base64
                    },
                    timeout=60.0  # больший таймаут для редактирования
                )

                if response.status_code == 200:
                    result = response.json()
                    image_data = result["image"]

                    # Декодируем base64 изображение и сохраняем в файл
                    image_bytes = base64.b64decode(image_data)
                    with open(output_file_path, "wb") as f:
                        f.write(image_bytes)

                    # Обновляем статус проекта на "completed" и сохраняем путь к файлу
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project:
                        project.status = ProjectStatus.completed
                        project.image_path = output_file_path
                        db.commit()

                    return True
                else:
                    # Если произошла ошибка при редактировании, ставим статус "failed"
                    project = db.query(Project).filter(Project.id == project_id).first()
                    if project:
                        project.status = ProjectStatus.failed
                        db.commit()
                    return False

        # Запускаем асинхронную функцию
        asyncio.run(edit_image_async())

    except Exception as e:
        # В случае ошибки обновляем статус на "failed"
        print(f"Error during image editing: {e}")
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            db.commit()


@router.post("/generate_element_images", response_model=GenerateImageResponse)
async def generate_element_images_endpoint(
    request: GenerateElementImageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Эндпоинт для генерации изображений для элементов сценария
    Принимает ID проекта и индекс элемента (или None для всех элементов)
    """
    # Проверяем, что проект существует
    project = db.query(Project).filter(Project.id == request.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Проверяем, что проект имеет сценарий
    if not project.result_path or not os.path.exists(project.result_path):
        raise HTTPException(status_code=404, detail="Scenario JSON not found")

    # Читаем сценарий из JSON файла
    with open(project.result_path, 'r', encoding='utf-8') as f:
        scenario_data = json.load(f)

    blocks = scenario_data.get('blocks', [])

    if request.element_index is not None:
        # Генерируем изображение только для указанного элемента
        if request.element_index < 1 or request.element_index > len(blocks):
            raise HTTPException(status_code=400, detail="Invalid element index")

        target_blocks = [blocks[request.element_index - 1]]
    else:
        # Генерируем изображения для всех элементов
        target_blocks = blocks

    # Определяем директорию для изображений
    user_data_dir = Path("api/users_data") / str(project.user_id) / str(project.id)
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # Обновляем статус проекта на "in_progress"
    project.status = ProjectStatus.in_progress
    db.commit()

    # Добавляем задачу в фон для генерации изображений для элементов
    background_tasks.add_task(
        process_element_images_generation,
        project.id,
        request.element_index,
        target_blocks,
        str(user_data_dir),
        db
    )

    if request.element_index is not None:
        message = f"Image generation started for element {request.element_index} of project {project.id}"
    else:
        message = f"Image generation started for all elements of project {project.id}"

    return GenerateImageResponse(
        project_id=project.id,
        status="in_progress",
        message=message
    )


def process_element_images_generation(
    project_id: int,
    element_index: Optional[int],
    target_blocks: list,
    user_data_dir: str,
    db: Session
):
    """
    Фоновая задача для генерации изображений для элементов сценария
    """
    try:
        import httpx
        import asyncio

        # Асинхронная функция для генерации изображений для элементов
        async def generate_element_images_async():
            success_count = 0

            for block in target_blocks:
                element_index = block.get('index')
                block_type = block.get('type')
                block_content = block.get('content', {})

                # Формируем описание изображения на основе типа и содержимого блока
                if block_type == 'scene_heading':
                    location = block_content.get('location', 'неизвестное место')
                    time = block_content.get('time', 'неизвестное время')
                    image_description = f"Сцена в {location} во время {time}. Съемка рекламного ролика."
                elif block_type == 'action':
                    description = block_content.get('description', '')
                    image_description = f"Действие: {description}"
                elif block_type == 'character':
                    name = block_content.get('name', 'Персонаж')
                    desc = block_content.get('description', '')
                    image_description = f"Портрет {name}. {desc}"
                elif block_type == 'dialogue':
                    speaker = block_content.get('speaker', 'Говорящий')
                    text = block_content.get('text', '')
                    image_description = f"Персонаж {speaker} говорит: '{text[:50]}...'"
                elif block_type == 'transition':
                    desc = block_content.get('description', '')
                    image_description = f"Переход: {desc}"
                else:
                    image_description = f"Сцена типа {block_type}: {str(block_content)}"

                # Создаем путь к изображению для этого элемента
                output_file_path = f"{user_data_dir}/{project_id}_{element_index}_element_image.png"

                # Создаем запись в базе данных для элемента
                element_image = ScenarioElementImage(
                    project_id=project_id,
                    element_index=element_index,
                    image_description=image_description,
                    status=ProjectStatus.in_progress
                )
                db.add(element_image)
                db.commit()

                try:
                    async with httpx.AsyncClient() as client:
                        response = await client.post(
                            "http://127.0.0.1:3339/generate_image",
                            json={"prompt": image_description},
                            timeout=30.0
                        )

                        if response.status_code == 200:
                            result = response.json()
                            image_data = result["image"]

                            # Декодируем base64 изображение и сохраняем в файл
                            image_bytes = base64.b64decode(image_data)
                            with open(output_file_path, "wb") as f:
                                f.write(image_bytes)

                            # Обновляем запись в базе данных
                            element_image.image_path = output_file_path
                            element_image.status = ProjectStatus.completed
                            db.commit()

                            success_count += 1
                        else:
                            # Обновляем статус элемента как failed
                            element_image.status = ProjectStatus.failed
                            db.commit()
                except Exception as e:
                    # В случае ошибки обновляем статус элемента как failed
                    element_image.status = ProjectStatus.failed
                    db.commit()
                    print(f"Error during image generation for element {element_index}: {e}")

            # После завершения всех генераций обновляем статус проекта
            project = db.query(Project).filter(Project.id == project_id).first()
            if project:
                # Проверяем, есть ли еще элементы в процессе генерации
                remaining_elements = db.query(ScenarioElementImage).filter(
                    ScenarioElementImage.project_id == project_id,
                    ScenarioElementImage.status == ProjectStatus.in_progress
                ).count()

                if remaining_elements == 0:
                    # Если все элементы обработаны, обновляем статус проекта
                    project.status = ProjectStatus.completed
                    db.commit()

        # Запускаем асинхронную функцию
        asyncio.run(generate_element_images_async())

    except Exception as e:
        # В случае ошибки обновляем статус проекта как failed
        print(f"Error during element images generation: {e}")
        project = db.query(Project).filter(Project.id == project_id).first()
        if project:
            project.status = ProjectStatus.failed
            db.commit()


@router.post("/edit_element_image", response_model=GenerateImageResponse)
async def edit_element_image_endpoint(
    request: EditElementImageRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    Эндпоинт для редактирования изображения конкретного элемента сценария
    Принимает ID проекта, индекс элемента и новое описание изображения
    """
    # Проверяем, что проект существует
    project = db.query(Project).filter(Project.id == request.project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    # Проверяем, что элемент изображения существует
    element_image = db.query(ScenarioElementImage).filter(
        ScenarioElementImage.project_id == request.project_id,
        ScenarioElementImage.element_index == request.element_index
    ).first()

    if not element_image:
        raise HTTPException(status_code=404, detail="Element image not found")

    # Проверяем, что у элемента есть изображение для редактирования
    if not element_image.image_path or not os.path.exists(element_image.image_path):
        raise HTTPException(status_code=404, detail="Original element image not found")

    # Обновляем статус элемента на "in_progress" и сохраняем новое описание
    element_image.status = ProjectStatus.in_progress
    element_image.image_description = request.image_description
    db.commit()

    # Обновляем также статус проекта как в процессе
    project.status = ProjectStatus.in_progress
    db.commit()

    # Добавляем задачу в фон для редактирования изображения элемента
    background_tasks.add_task(
        process_element_image_editing,
        element_image.id,
        request.image_description,
        element_image.image_path,  # путь к оригинальному изображению
        db
    )

    return GenerateImageResponse(
        project_id=project.id,
        status="in_progress",
        message=f"Element image editing started for element {request.element_index} of project {project.id}"
    )


def process_element_image_editing(
    element_image_id: int,
    image_description: str,
    original_image_path: str,
    db: Session
):
    """
    Фоновая задача для редактирования изображения конкретного элемента
    """
    try:
        import httpx
        import asyncio
        import base64

        # Асинхронная функция для редактирования изображения элемента
        async def edit_element_image_async():
            # Читаем оригинальное изображение и кодируем в base64
            with open(original_image_path, "rb") as f:
                original_image_bytes = f.read()
            original_image_base64 = base64.b64encode(original_image_bytes).decode('utf-8')

            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "http://127.0.0.1:3339/edit_image",
                    json={
                        "prompt": image_description,
                        "image_base64": original_image_base64
                    },
                    timeout=60.0  # больший таймаут для редактирования
                )

                if response.status_code == 200:
                    result = response.json()
                    image_data = result["image"]

                    # Новый путь к изображению (тот же файл, просто перезаписываем)
                    element_image = db.query(ScenarioElementImage).filter(
                        ScenarioElementImage.id == element_image_id
                    ).first()

                    if element_image:
                        # Декодируем base64 изображение и сохраняем в файл
                        image_bytes = base64.b64decode(image_data)
                        with open(element_image.image_path, "wb") as f:
                            f.write(image_bytes)

                        # Обновляем статус элемента на "completed"
                        element_image.status = ProjectStatus.completed
                        db.commit()

                        # Также проверяем, все ли элементы проекта завершены
                        project = db.query(Project).filter(Project.id == element_image.project_id).first()
                        if project:
                            remaining_elements = db.query(ScenarioElementImage).filter(
                                ScenarioElementImage.project_id == element_image.project_id,
                                ScenarioElementImage.status == ProjectStatus.in_progress
                            ).count()

                            if remaining_elements == 0:
                                # Если все элементы обработаны, обновляем статус проекта
                                project.status = ProjectStatus.completed
                                db.commit()

                    return True
                else:
                    # Если произошла ошибка при редактировании, ставим статус "failed"
                    element_image = db.query(ScenarioElementImage).filter(
                        ScenarioElementImage.id == element_image_id
                    ).first()

                    if element_image:
                        element_image.status = ProjectStatus.failed
                        db.commit()

                    return False

        # Запускаем асинхронную функцию
        asyncio.run(edit_element_image_async())

    except Exception as e:
        # В случае ошибки обновляем статус элемента как failed
        print(f"Error during element image editing: {e}")
        element_image = db.query(ScenarioElementImage).filter(
            ScenarioElementImage.id == element_image_id
        ).first()
        if element_image:
            element_image.status = ProjectStatus.failed
            db.commit()


@router.get("/status/{project_id}", response_model=ProjectStatusResponse)
def get_project_status(
    project_id: int,
    db: Session = Depends(get_db)
):
    """
    Получить статус проекта по ID (включая информацию о сценарии и изображении)
    """
    project = db.query(Project).filter(Project.id == project_id).first()
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")

    return ProjectStatusResponse(
        project_id=project.id,
        user_id=project.user_id,
        status=project.status.value,
        created_at=project.created_at,
        updated_at=project.updated_at,
        result_path=project.result_path,
        image_path=project.image_path,
        image_description=project.image_description
    )