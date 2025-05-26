# src/api/main.py
from fastapi import FastAPI, HTTPException, Query, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import pandas as pd
import pickle
import os
import sys
from fastapi.middleware.cors import CORSMiddleware


# Add src to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.recommendation_model import RecommendationEngine

# Load the trained model
def load_model():
    model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models'))
    model_files = [f for f in os.listdir(model_dir) if f.startswith('recommendation_model_') and f.endswith('.pkl')]
    
    if not model_files:
        # If no trained model exists, return None
        return None
        
    # Get the most recent model
    latest_model = sorted(model_files)[-1]
    model_path = os.path.join(model_dir, latest_model)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    return model

# Define request and response models
class SkillRequirement(BaseModel):
    skill_name: str
    importance: float = Field(1.0, ge=0.0, le=1.0)

class RoleRequirement(BaseModel):
    role_name: str
    count: int = Field(1, ge=1)

class ProjectRequirement(BaseModel):
    project_id: Optional[str] = None
    project_name: str
    required_skills: List[SkillRequirement] = []
    required_roles: List[RoleRequirement] = []
    existing_team: List[str] = []
    team_size: int = Field(5, ge=1, le=20)

class TeamMember(BaseModel):
    resource_id: str
    score: float
    explanation: Optional[str] = None

class TeamRecommendation(BaseModel):
    team: List[TeamMember]
    project_name: str
    team_size: int

# Initialize FastAPI app
app = FastAPI(
    title="Float Resource Recommendation API",
    description="API for recommending optimal team compositions for projects",
    version="0.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for the model
def get_model():
    model = load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not trained yet. Please train the model first.")
    return model

@app.get("/")
async def root():
    return {"message": "Welcome to Float Resource Recommendation API"}

@app.get("/health")
async def health_check(model: RecommendationEngine = Depends(get_model)):
    return {"status": "healthy", "model_trained": model.is_trained}

@app.post("/recommend", response_model=TeamRecommendation)
async def recommend_team(project: ProjectRequirement, model: RecommendationEngine = Depends(get_model)):
    # Transform project requirements to the format expected by the model
    required_skills = {skill.skill_name: skill.importance for skill in project.required_skills}
    required_roles = {role.role_name: role.count for role in project.required_roles}
    
    project_requirements = {
        'id': project.project_id,
        'name': project.project_name,
        'required_skills': required_skills,
        'required_roles': required_roles
    }
    
    # Get recommendations
    try:
        recommendations = model.recommend_team(
            project_requirements,
            team_size=project.team_size,
            existing_team=project.existing_team
        )
        
        # Get explanations
        explanations = model.get_recommendation_explanations(recommendations, project_requirements)
        
        # Format response
        team = []
        for resource_id, score in recommendations.items():
            explanation = explanations.get(resource_id, "No explanation available.")
            team.append(TeamMember(
                resource_id=resource_id,
                score=float(score),
                explanation=explanation
            ))
            
        return TeamRecommendation(
            team=team,
            project_name=project.project_name,
            team_size=len(team)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)