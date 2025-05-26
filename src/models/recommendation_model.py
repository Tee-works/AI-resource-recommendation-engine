# src/models/recommendation_model.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from src.models.collaborative_filtering import CollaborativeFilteringModel
from src.models.content_based_filtering import ContentBasedFilteringModel

class RecommendationEngine:
    """Main recommendation engine combining multiple recommendation models."""
    
    def __init__(self):
        """Initialize the recommendation engine."""
        self.collaborative_model = CollaborativeFilteringModel()
        self.content_based_model = ContentBasedFilteringModel()
        self.is_trained = False
        
    def train(self, data: Dict[str, pd.DataFrame]):
        """
        Train the recommendation engine with the provided data.
        
        Args:
            data: Dictionary containing processed datasets and feature matrices.
        """
        # Train collaborative filtering model
        if 'project_resource_matrix' in data and 'project_success' in data:
            self.collaborative_model.train(
                data['project_resource_matrix'],
                data['project_success']
            )
            
        # Train content-based filtering model
        if 'resource_skill_matrix' in data:
            resource_metadata = data.get('resource_metadata')
            self.content_based_model.train(
                data['resource_skill_matrix'],
                resource_metadata
            )
            
        self.is_trained = True
        
    def recommend_team(self, project_requirements: Dict, team_size: int = 5,
                      existing_team: List[str] = None,
                      collaborative_weight: float = 0.4,
                      content_based_weight: float = 0.6) -> pd.Series:
        """
        Recommend a team for a project by combining different recommendation models.
        
        Args:
            project_requirements: Dictionary containing project requirements.
            team_size: Number of team members to recommend.
            existing_team: List of resource IDs already assigned to the project.
            collaborative_weight: Weight for collaborative filtering model.
            content_based_weight: Weight for content-based filtering model.
            
        Returns:
            Series with resource IDs as index and recommendation scores as values.
        """
        if not self.is_trained:
            raise ValueError("Models have not been trained. Call train() first.")
            
        # Initialize empty team if none provided
        if existing_team is None:
            existing_team = []
            
        # Get recommendations from collaborative model
        collaborative_recs = None
        if self.collaborative_model.is_trained:
            collaborative_recs = self.collaborative_model.recommend_team(
                project_requirements,
                team_size=team_size*2,  # Get more recommendations to ensure overlap
                existing_team=existing_team
            )
            
        # Get recommendations from content-based model
        content_based_recs = None
        if self.content_based_model.is_trained:
            content_based_recs = self.content_based_model.recommend_team(
                project_requirements,
                team_size=team_size*2,
                existing_team=existing_team
            )
            
        # Combine recommendations
        if collaborative_recs is not None and content_based_recs is not None:
            # Find common resource IDs
            common_resources = set(collaborative_recs.index).intersection(set(content_based_recs.index))
            
            # Calculate weighted scores for common resources
            combined_scores = pd.Series(index=list(common_resources))
            
            for resource_id in common_resources:
                combined_scores[resource_id] = (
                    collaborative_weight * collaborative_recs[resource_id] +
                    content_based_weight * content_based_recs[resource_id]
                )
                
            # Sort and return top recommendations
            return combined_scores.sort_values(ascending=False).head(team_size)
            
        # If one model is not trained, return recommendations from the other
        elif collaborative_recs is not None:
            return collaborative_recs.head(team_size)
        elif content_based_recs is not None:
            return content_based_recs.head(team_size)
            
        # If no models are trained, return empty series
        return pd.Series()
        
    def get_recommendation_explanations(self, recommended_team: pd.Series, 
                                      project_requirements: Dict) -> Dict[str, str]:
        """
        Get explanations for why each team member was recommended.
        
        Args:
            recommended_team: Series with resource IDs as index and recommendation scores as values.
            project_requirements: Dictionary containing project requirements.
            
        Returns:
            Dictionary mapping resource IDs to explanation strings.
        """
        explanations = {}
        
        # Placeholder implementation - in a real system, this would extract actual insights
        # from the recommendation models to explain why each resource was recommended
        
        for resource_id, score in recommended_team.items():
            explanations[resource_id] = (
                f"Resource {resource_id} was recommended with a score of {score:.2f}. "
                f"This resource has a good match with the project requirements based on "
                f"skills, role, and past performance on similar projects."
            )
            
        return explanations