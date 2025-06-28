import os
from dotenv import load_dotenv
from fastapi.middleware.wsgi import WSGIMiddleware
import uvicorn
from app.api import app

load_dotenv()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    PORT = int(os.getenv("PORT", 8000))
    HOST = os.getenv("HOST", "0.0.0.0")
    print(f"Server running at http://{HOST}:{PORT}")
    uvicorn.run(
        app="app.api:app",
        host=HOST, port=PORT
    )
