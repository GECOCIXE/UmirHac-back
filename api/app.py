from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from api.routers import auth, script_generator, folders

from jose import jwt, JWTError
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
ALGORITHM = "HS256"

# Настройка авторизации для Swagger
security = HTTPBearer(description="Введите токен JWT в формате: Bearer &lt;token&gt;")

app = FastAPI(
    title="UmirHac API",
    description="API для генерации сценариев и изображений",
    version="1.0.0",
    docs_url="/docs",
    openapi_url="/openapi.json",
    swagger_ui_parameters={"defaultModelsExpandDepth": -1}
)

# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Разрешить все источники
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

PUBLIC_PATHS = {
    "/auth/login",
    "/auth/register",
    "/docs",
    "/openapi.json"
}


def _strip_bearer(auth_header: str | None):
    if not auth_header:
        return None
    if auth_header.startswith("Bearer "):
        return auth_header[len("Bearer "):]
    return None


@app.middleware("http")
async def jwt_auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
        return await call_next(request)

    raw = request.headers.get("Authorization")
    token = _strip_bearer(raw)
    if not token:
        return JSONResponse(status_code=401, content={"detail": "Authorization header is missing or invalid"})

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp = payload.get("exp")
        if exp is not None and datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(tz=timezone.utc):
            return JSONResponse(status_code=401, content={"detail": "Token expired"})
        username: str = payload.get("sub")
        if username is None:
            raise JWTError("No sub in token")
        print(f"Middleware: decoded token, username/sub: {username}")
        request.state.user = username
    except JWTError as e:
        print(f"JWT Error: {e}")
        return JSONResponse(status_code=401, content={"detail": "Token is invalid or expired"})

    return await call_next(request)

app.include_router(auth.router)
app.include_router(script_generator.router)
app.include_router(folders.router)