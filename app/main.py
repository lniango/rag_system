# Developing the API with FastAPI
from fastapi import FastAPI
from app.endpoints import router

# setting up a FastAPI instance, and then pulling in all the routes
# (or API paths) from another file, which we'll create next
app = FastAPI()
app.include_router(router)

