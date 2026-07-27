"""
modal_app.py 
- deploys the existing app.py to Modal.


Run locally with: modal serve modal_app.py
Deploy with:      modal deploy modal_app.py
"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.13")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(
        ".",
        remote_path="/app",
        ignore=[
            "geo_env",              # venv
            "train.csv",            # training data
            "embeddings.npz",       # training embeddings
            "osv5m_filtered",       # training data folder
            "model_versions",       # local model checkpoints
            "checking",             # misc python files
            "confusion_matrix.npy", 
            "classes.pkl",          
            "corrections.pkl",     
            "__pycache__",
            "*.pyc",
            ".git",
            ".DS_Store",
            ".env", 
        ],
    )
)

app = modal.App("geocoach", image=image)


@app.function(
    secrets=[modal.Secret.from_name("geocoach-secrets")],
    min_containers=0, 
    max_containers=1, 
    timeout=120,   # timeout for model loading & inference
)
@modal.concurrent(max_inputs=5)
@modal.wsgi_app()
def flask_app():
    import sys
    sys.path.insert(0, "/app")
    from app import app as flask_application
    return flask_application