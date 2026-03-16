# Copyright (c) 2026, Salesforce, Inc.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/Apache-2.0
"""
Vertex AI Gemini Fine-tuning Script
====================================
Fine-tunes Gemini 1.5 Flash on the Merlion tool-calling dataset.

Prerequisites:
1. gcloud CLI installed and authenticated
2. Google Cloud project with Vertex AI enabled
3. training_data.jsonl file generated

Usage:
    python finetune_vertex.py

Cost: ~$10-15 for 100 examples, 3 epochs
"""
import os
import json
from pathlib import Path


def check_prerequisites():
    """Check if prerequisites are met."""
    print("🔍 Checking prerequisites...\n")
    
    # Check training data exists
    if not Path("training_data.jsonl").exists():
        print("❌ training_data.jsonl not found. Run: python generate_training_data.py")
        return False
    
    # Count examples
    with open("training_data.jsonl") as f:
        n_examples = sum(1 for _ in f)
    print(f"✅ Training data: {n_examples} examples")
    
    # Check gcloud CLI
    import subprocess
    try:
        result = subprocess.run(["gcloud", "--version"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ gcloud CLI not installed or not in PATH")
            return False
        print("✅ gcloud CLI available")
    except FileNotFoundError:
        print("❌ gcloud CLI not found. Install from: https://cloud.google.com/sdk/docs/install")
        return False
    
    # Check authentication
    result = subprocess.run(["gcloud", "auth", "list", "--format=json"], capture_output=True, text=True)
    if result.returncode != 0 or "[]" in result.stdout:
        print("❌ Not authenticated. Run: gcloud auth login")
        return False
    print("✅ gcloud authenticated")
    
    # Check project
    result = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True)
    project = result.stdout.strip()
    if not project or project == "(unset)":
        print("❌ No project set. Run: gcloud config set project YOUR_PROJECT_ID")
        return False
    print(f"✅ Project: {project}")
    
    return True


def print_cost_estimate():
    """Print cost estimate before proceeding."""
    print("\n" + "="*60)
    print("💰 COST ESTIMATE")
    print("="*60)
    print("""
Model: Gemini 1.5 Flash (tuning via Vertex AI)
Examples: 100
Epochs: 3
Estimated Time: ~30 minutes
Estimated Cost: ~$10-15 from your Google Cloud credits

After tuning completes:
- Inference cost: $0.075 per 1M input tokens
- This is the same as the base model
""")
    print("="*60)


def print_manual_instructions():
    """Print manual instructions for Vertex AI tuning."""
    print("""
📋 MANUAL VERTEX AI TUNING INSTRUCTIONS
========================================

Since Vertex AI tuning requires Google Cloud Console access, 
here are the steps to follow:

1. Go to: https://console.cloud.google.com/vertex-ai/tuning

2. Click "Create tuned model"

3. Settings:
   - Base model: gemini-1.5-flash-002
   - Display name: merlion-tool-caller
   - Upload training data: training_data.jsonl (from this folder)
   - Epochs: 3 (or leave default)
   
4. Click "Start tuning"

5. Wait ~30 minutes for completion

6. Once done, copy the tuned model endpoint name (looks like):
   projects/YOUR_PROJECT/locations/us-central1/endpoints/ENDPOINT_ID

7. Update this project to use the tuned model (see below)

========================================

ALTERNATIVE: Use Python SDK (requires setup)

```python
from google.cloud import aiplatform

aiplatform.init(project="YOUR_PROJECT", location="us-central1")

# Create tuning job
tuning_job = aiplatform.SFTTuningJob.create(
    source_model="gemini-1.5-flash-002",
    training_data="gs://YOUR_BUCKET/training_data.jsonl",
    tuned_model_display_name="merlion-tool-caller",
    epochs=3,
)

# Wait for completion
tuning_job.wait()
print(f"Tuned model: {tuning_job.tuned_model_endpoint_name}")
```

========================================

After tuning, update llm.py to use the tuned model:

```python
# In llm.py, update the model name:
class GeminiLLM:
    def __init__(self, model="YOUR_TUNED_MODEL_ENDPOINT"):
        ...
```
""")


def upload_to_gcs():
    """Upload training data to Google Cloud Storage."""
    import subprocess
    
    # Get project ID
    result = subprocess.run(["gcloud", "config", "get-value", "project"], capture_output=True, text=True)
    project = result.stdout.strip()
    
    bucket_name = f"{project}-merlion-tuning"
    
    print(f"\n📤 Uploading training data to GCS...")
    
    # Create bucket if needed
    subprocess.run([
        "gsutil", "mb", "-l", "us-central1", f"gs://{bucket_name}"
    ], capture_output=True)
    
    # Upload file
    result = subprocess.run([
        "gsutil", "cp", "training_data.jsonl", f"gs://{bucket_name}/training_data.jsonl"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        gcs_path = f"gs://{bucket_name}/training_data.jsonl"
        print(f"✅ Uploaded to: {gcs_path}")
        return gcs_path
    else:
        print(f"❌ Upload failed: {result.stderr}")
        return None


if __name__ == "__main__":
    print("🚀 Vertex AI Gemini Fine-tuning Setup\n")
    
    prereqs_ok = check_prerequisites()
    
    print_cost_estimate()
    
    if prereqs_ok:
        response = input("\n🔄 Upload training data to GCS? (y/n): ").strip().lower()
        if response == 'y':
            gcs_path = upload_to_gcs()
            if gcs_path:
                print(f"\n✅ Ready for tuning! Use this GCS path: {gcs_path}")
    
    print_manual_instructions()
