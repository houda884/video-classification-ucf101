"""
MLflow configuration and utilities
"""
import mlflow
import mlflow.keras
import mlflow.tensorflow
from datetime import datetime
import os

def setup_mlflow(experiment_name="video_classification"):
    """Setup MLflow tracking"""
    # Create mlruns directory if it doesn't exist
    os.makedirs("mlruns", exist_ok=True)
    
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            name=experiment_name,
            tags={"project": "video-classification", "type": "research"}
        )
    else:
        experiment_id = experiment.experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    return experiment_id

def log_training_params(params):
    """Log training parameters to MLflow"""
    for key, value in params.items():
        mlflow.log_param(key, value)

def log_training_metrics(metrics, epoch=None):
    """Log training metrics to MLflow"""
    for key, value in metrics.items():
        if epoch is not None:
            mlflow.log_metric(key, value, step=epoch)
        else:
            mlflow.log_metric(key, value)

def log_model(model, model_name, artifact_path="models"):
    """Log model to MLflow"""
    mlflow.keras.log_model(
        model,
        artifact_path=artifact_path,
        registered_model_name=model_name
    )

def register_best_model(run_id, model_name="tamper_detection_best"):
    """Register best model from run"""
    client = mlflow.tracking.MlflowClient()
    
    # Get best run
    runs = client.search_runs(
        experiment_ids=["0"],
        filter_string="metrics.val_accuracy > 0.85",
        order_by=["metrics.val_accuracy DESC"],
        max_results=1
    )
    
    if runs:
        best_run = runs[0]
        model_uri = f"runs:/{best_run.info.run_id}/model"
        
        # Register model
        mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f"✅ Registered best model: {model_name}")
        return True
    
    return False

def get_model_version(model_name, stage="Production"):
    """Get model version for deployment"""
    client = mlflow.tracking.MlflowClient()
    
    try:
        model_versions = client.get_latest_versions(model_name, stages=[stage])
        if model_versions:
            return model_versions[0]
    except:
        pass
    
    return None