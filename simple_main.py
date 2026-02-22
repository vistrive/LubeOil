from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI(title="LOBP Control System", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "LOBP Control System API", "status": "running"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/api/v1/health")
def api_health():
    return {"status": "healthy", "service": "api"}

@app.get("/api/v1/recipes")
def get_recipes():
    return {
        "recipes": [
            {"id": 1, "name": "SAE 10W-40", "status": "active"},
            {"id": 2, "name": "SAE 15W-40", "status": "active"},
        ]
    }

@app.get("/api/v1/tanks")
def get_tanks():
    return {
        "tanks": [
            {"id": 1, "name": "Tank T-101", "level": 85, "capacity": 1000},
            {"id": 2, "name": "Tank T-102", "level": 45, "capacity": 1000},
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)