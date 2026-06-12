from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger

from api.routes import cameras, models, events, health, webhooks
from api.websocket.feed import router as ws_router
from core.events.bus import event_bus


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("VisionSync starting up...")
    await event_bus.start()
    yield
    logger.info("VisionSync shutting down...")
    await event_bus.stop()


app = FastAPI(
    title="VisionSync",
    description="Universal AI vision platform — plug any camera in, get intelligence out.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# REST routes
app.include_router(health.router, tags=["health"])
app.include_router(cameras.router, prefix="/api/cameras", tags=["cameras"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(events.router, prefix="/api/events", tags=["events"])
app.include_router(webhooks.router, prefix="/api/webhooks", tags=["webhooks"])

# WebSocket
app.include_router(ws_router, prefix="/ws", tags=["websocket"])