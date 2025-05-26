# src/models/collaborative_filtering.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

class CollaborativeFilteringModel:
    """Collaborative filtering model for team recommendations."""
    
    def __init__(self):
        """Initialize the collaborative filtering model."""
        self.project_resource_matrix = None
        self.resource_project_matrix = None
        self.project_similarity = None
        self.resource_similarity = None
        self.project_success = None
        self.is_trained = False
        
    def train(self, project_resource_matrix: pd.DataFrame, project_success: Optional[pd.DataFrame] = None):
        """
        Train the collaborative filtering model.
        
        Args:
            project_resource_matrix: DataFrame with project IDs as index, resource IDs as columns,
                                     and allocation weights as values.
            project_success: Optional DataFrame with project IDs as index and success scores.
        """
        self.project_resource_matrix = project_resource_matrix
        
        # Transpose to get resource-project matrix
        self.resource_project_matrix = project_resource_matrix.T
        
        # Calculate project similarity based on resource allocations
        self.project_similarity = pd.DataFrame(
            cosine_similarity(project_resource_matrix),
            index=project_resource_matrix.index,
            columns=project_resource_matrix.index
        )
        
        # Calculate resource similarity based on project allocations
        self.resource_similarity = pd.DataFrame(
            cosine_similarity(self.resource_project_matrix),
            index=self.resource_project_matrix.index,
            columns=self.resource_project_matrix.index
        )
        
        # Store project success data if provided
        if project_success is not None:
            self.project_success = project_success
            
        self.is_trained = True
        
    def recommend_team(self, project_requirements: Dict, team_size: int = 5,
                      existing_team: List[str] = None) -> pd.Series:
        """
        Recommend team members for a project.
        
        Args:
            project_requirements: Dictionary containing project requirements.
            team_size: Number of team members to recommend.
            existing_team: List of resource IDs already assigned to the project.
            
        Returns:
            Series with resource IDs as index and recommendation scores as values.
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained. Call train() first.")
            
        # Initialize empty team if none provided
        if existing_team is None:
            existing_team = []
            
        # Find similar projects
        similar_projects = self._find_similar_projects(project_requirements)
        
        # Get resources from similar projects, weighted by project similarity and success
        resource_scores = self._score_resources_from_similar_projects(similar_projects, existing_team)
        
        # If we have resource similarity data, adjust scores based on fit with existing team
        if existing_team and len(existing_team) > 0:
            resource_scores = self._adjust_scores_for_team_fit(resource_scores, existing_team)
            
        # Sort and return top recommendations
        return resource_scores.sort_values(ascending=False).head(team_size)
        
    def _find_similar_projects(self, project_requirements: Dict) -> pd.Series:
        """
        Find projects similar to the given requirements.
        
        Args:
            project_requirements: Dictionary containing project requirements.
            
        Returns:
            Series with project IDs as index and similarity scores as values.
        """
        # For a new project, we need to understand what makes it similar to existing projects
        # This could be based on various factors like required skills, budget, duration, etc.
        # For now, we'll implement a simple approach using role requirements if provided
        
        if 'required_roles' in project_requirements:
            required_roles = project_requirements['required_roles']
            
            # Create a resource preference vector based on roles
            resource_preferences = pd.Series(0, index=self.resource_project_matrix.index)
            
            # Here we'd look up resources by role and assign preference weights
            # For the portfolio project, we'll just use a placeholder implementation
            # In a real system, this would use actual resource metadata
            
            for role, importance in required_roles.items():
                # Assume we have a lookup of resources by role (placeholder)
                resources_with_role = self.resource_project_matrix.index[:5]  # Just use first 5 as placeholder
                resource_preferences[resources_with_role] += importance
                
            # Use this preference vector to find similar projects
            similar_projects = self.project_resource_matrix.dot(resource_preferences)
            
            # Normalize
            similar_projects = similar_projects / similar_projects.max()
            
            return similar_projects
            
        # If no role requirements, use a more generic approach based on project attributes
        # For now, just return equal weights for all projects as a placeholder
        return pd.Series(1, index=self.project_resource_matrix.index)
        
    def _score_resources_from_similar_projects(self, project_similarities: pd.Series,
                                              existing_team: List[str]) -> pd.Series:
        """
        Score resources based on their involvement in similar projects.
        
        Args:
            project_similarities: Series with project IDs as index and similarity scores as values.
            existing_team: List of resource IDs already assigned to the project.
            
        Returns:
            Series with resource IDs as index and recommendation scores as values.
        """
        # Initialize scores for all resources
        resource_scores = pd.Series(0, index=self.resource_project_matrix.index)
        
        # Exclude existing team members
        available_resources = [r for r in resource_scores.index if r not in existing_team]
        
        # For each similar project
        for project_id, similarity in project_similarities.items():
            if similarity <= 0:
                continue
                
            # Get the resources that worked on this project
            project_resources = self.project_resource_matrix.loc[project_id]
            project_resources = project_resources[project_resources > 0]
            
            # Weight by similarity and success (if available)
            weight = similarity
            if self.project_success is not None and project_id in self.project_success.index:
                weight *= self.project_success.loc[project_id, 'success_score']
                
            # Add weighted scores to resource scores
            for resource_id, allocation in project_resources.items():
                if resource_id in available_resources:
                    resource_scores[resource_id] += weight * allocation
                    
        # Normalize scores
        if resource_scores.max() > 0:
            resource_scores = resource_scores / resource_scores.max()
            
        return resource_scores
        
    def _adjust_scores_for_team_fit(self, resource_scores: pd.Series, existing_team: List[str]) -> pd.Series:
        """
        Adjust resource scores based on how well they fit with the existing team.
        
        Args:
            resource_scores: Series with resource IDs as index and recommendation scores as values.
            existing_team: List of resource IDs already assigned to the project.
            
        Returns:
            Adjusted resource scores.
        """
        # For each potential team member, calculate their average similarity to the existing team
        adjusted_scores = resource_scores.copy()
        
        for resource_id in resource_scores.index:
            if resource_id in existing_team:
                continue
                
            # Calculate average similarity to existing team
            team_similarities = []
            for team_member in existing_team:
                if team_member in self.resource_similarity.index and resource_id in self.resource_similarity.columns:
                    team_similarities.append(self.resource_similarity.loc[team_member, resource_id])
                    
            if team_similarities:
                avg_similarity = sum(team_similarities) / len(team_similarities)
                # Adjust score to balance individual score and team fit
                adjusted_scores[resource_id] = 0.7 * resource_scores[resource_id] + 0.3 * avg_similarity
                
        return adjusted_scores