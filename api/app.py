from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from api.routers import auth, script_generator

app = FastAPI()
# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все источники
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(script_generator.router)
