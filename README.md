# Heart Disease Prediction (MLOps) — End-to-End Project
This project trains a heart-disease classifier and serves it via a FastAPI `/predict` endpoint.
It supports running locally, via Docker, and via Kubernetes (Docker Desktop Kubernetes or Minikube).

---

## Prerequisites
Install these before starting:
- Python 3.10+
- pip
- Docker Desktop (recommended) OR Minikube
- kubectl
- VS Code (optional but recommended)

Check versions:
```bash
python --version
pip --version
docker --version
kubectl version --client


heart-disease-mlops/
├─ app.py
├─ requirements.txt
├─ Dockerfile
├─ .dockerignore
├─ sample_request.json
├─ heart_disease_pipeline.joblib        <-- generated from notebook 
├─ MLOps_Assignment.ipynb               <-- your notebook
└─ k8s/
   ├─ deployment.yaml
   └─ service.yaml


## Build image
1.1 Build image

From project root:

cd heart-disease-mlops
docker build -t heart-api:1 .

1.2 Run container
docker run -p 8000:8000 heart-api:1

1.3 Test health
curl http://127.0.0.1:8000/health

1.4 Test predict
curl -s -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d @sample_request.json


Stop container: Ctrl + C

2) Run on Kubernetes (Required for deployment marks)

You can deploy locally using either:

A) Docker Desktop Kubernetes (easiest), or

B) Minikube

Common: Kubernetes manifests

Make sure you have these files:

k8s/deployment.yaml

k8s/service.yaml

2A) Kubernetes using Docker Desktop (Recommended)
2A.1 Enable Kubernetes in Docker Desktop

Docker Desktop → Settings → Kubernetes → Enable Kubernetes → Apply & Restart

Confirm:

kubectl get nodes

2A.2 Build Docker image (normal build works)
cd heart-disease-mlops
docker build -t heart-api:1 .

2A.3 Deploy to Kubernetes
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

2A.4 Verify pods and service
kubectl get pods
kubectl get svc

2A.5 Port-forward service to test quickly
kubectl port-forward svc/heart-api-svc 8000:80


Now test from a new terminal:

curl http://127.0.0.1:8000/health
curl -s -X POST "http://127.0.0.1:8000/predict" -H "Content-Type: application/json" -d @sample_request.json


Stop port-forward: Ctrl + C

2B) Kubernetes using Minikube
2B.1 Start Minikube
minikube start
kubectl get nodes

2B.2 Build the Docker image INSIDE Minikube

This is important; otherwise K8s won’t find heart-api:1.

Mac/Linux:

eval $(minikube -p minikube docker-env)


Windows (PowerShell):

minikube -p minikube docker-env | Invoke-Expression


Now build:

cd heart-disease-mlops
docker build -t heart-api:1 .

2B.3 Deploy manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

2B.4 Check status
kubectl get pods
kubectl get svc

2B.5 Get service URL and test
minikube service heart-api-svc --url


Copy the printed URL and test:

# replace <URL> with the output from minikube
curl <URL>/health
curl -s -X POST "<URL>/predict" -H "Content-Type: application/json" -d @sample_request.json