from fastapi import FastAPI

from endpoints import router
from rag import lifespan

app = FastAPI(lifespan=lifespan)
app.include_router(router)
