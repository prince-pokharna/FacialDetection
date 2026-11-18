"""
Main FastAPI Application Entrypoint.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.core.logging import setup_logging, logger
from backend.app.core.security import get_password_hash
from backend.app.db.base import Base
from backend.app.db.session import engine, SessionLocal
from backend.app.models.user import User
from backend.app.services.biometric_service import biometric_service

from backend.app.api.auth import router as auth_router
from backend.app.api.people import router as people_router
from backend.app.api.recognize import router as recognize_router
from backend.app.api.attendance import router as attendance_router
from backend.app.api.system import router as system_router
from backend.app.api.websocket import router as ws_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize logging, database tables, seed initial admin, preload vector gallery
    setup_logging()
    logger.info("Starting up %s (Env: %s)...", settings.APP_NAME, settings.ENVIRONMENT)
    Base.metadata.create_all(bind=engine)

    # Seed default admin if missing
    db = SessionLocal()
    try:
        admin_user = db.query(User).filter(User.email == "admin@attendance.ai").first()
        if not admin_user:
            admin_user = User(
                email="admin@attendance.ai",
                hashed_password=get_password_hash("Admin@123456"),
                full_name="System Administrator",
                role="admin",
                is_active=True,
            )
            db.add(admin_user)
            db.commit()
            logger.info("Default administrator account seeded: admin@attendance.ai")

        # Load enrolled gallery into matcher
        biometric_service.load_gallery(db)
    finally:
        db.close()

    yield

    # Shutdown
    logger.info("Shutting down %s...", settings.APP_NAME)


app = FastAPI(
    title=settings.APP_NAME,
    description="Production-grade biometric open-set face recognition and attendance management platform.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS Middleware
origins = settings.CORS_ORIGINS if isinstance(settings.CORS_ORIGINS, list) else [settings.CORS_ORIGINS]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if "*" not in origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(auth_router)
app.include_router(people_router)
app.include_router(recognize_router)
app.include_router(attendance_router)
app.include_router(system_router)
app.include_router(ws_router)


@app.get("/")
def root():
    return {
        "app": settings.APP_NAME,
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }
