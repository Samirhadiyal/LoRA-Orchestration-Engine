from fastapi import FastAPI

app = FastAPI(title="NeuroMesh API Gateway")

@app.get("/")
def read_root():
    return {"status": "online", "message": "NeuroMesh API Gateway is running!"}