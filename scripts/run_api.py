# scripts/run_api.py
import os
import sys

# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_api():
    """Run the FastAPI server."""
    import uvicorn
    
    print("Starting Resource Recommendation API...")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    run_api()