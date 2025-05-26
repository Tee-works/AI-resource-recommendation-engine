# src/models/content_based_filtering.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedFilteringModel:
    """Content-based filtering model for team recommendations based on skills and attributes."""
    
    def __init__(self):
        """Initialize the content-based filtering model."""
        self.resource_skill_matrix = None
        self.resource_metadata = None
        self.is_trained = False
        
    def train(self, resource_skill_matrix: pd.DataFrame, resource_metadata: Optional[pd.DataFrame] = None):
        """
        Train the content-based filtering model.
        
        Args:
            resource_skill_matrix: DataFrame with resource IDs as index, skills as columns,
                                   and proficiency levels as values.
            resource_metadata: Optional DataFrame with resource metadata (role, department, etc.)
        """
        self.resource_skill_matrix = resource_skill_matrix
        self.resource_metadata = resource_metadata
        self.is_trained = True
        
    def recommend_team(self, project_requirements: Dict, team_size: int = 5,
                      existing_team: List[str] = None) -> pd.Series:
        """
        Recommend team members for a project based on content similarity.
        
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
            
        # Calculate skill match scores
        skill_scores = self._calculate_skill_match(project_requirements)
        
        # Calculate role match scores if role requirements provided
        role_scores = None
        if 'required_roles' in project_requirements and self.resource_metadata is not None:
            role_scores = self._calculate_role_match(project_requirements['required_roles'])
            
        # Combine scores
        final_scores = self._combine_scores(skill_scores, role_scores)
        
        # Remove existing team members
        for resource_id in existing_team:
            if resource_id in final_scores.index:
                final_scores.drop(resource_id, inplace=True)
                
        # Sort and return top recommendations
        return final_scores.sort_values(ascending=False).head(team_size)
        
    def _calculate_skill_match(self, project_requirements: Dict) -> pd.Series:
        """
        Calculate match scores based on skill requirements.
        
        Args:
            project_requirements: Dictionary containing project requirements.
            
        Returns:
            Series with resource IDs as index and skill match scores as values.
        """
        # Extract skill requirements if available
        required_skills = project_requirements.get('required_skills', {})
        
        if not required_skills:
            # If no skill requirements provided, return equal scores for all resources
            return pd.Series(1.0, index=self.resource_skill_matrix.index)
            
        # Create a skill requirement vector
        skill_vector = pd.Series(0, index=self.resource_skill_matrix.columns)
        
        for skill, importance in required_skills.items():
            if skill in skill_vector.index:
                skill_vector[skill] = importance
                
        # Calculate similarity between resource skills and requirements
        # Using dot product to weight by both proficiency and importance
        match_scores = self.resource_skill_matrix.dot(skill_vector)
        
        # Normalize scores
        if match_scores.max() > 0:
            match_scores = match_scores / match_scores.max()
            
        return match_scores
        
    def _calculate_role_match(self, required_roles: Dict) -> pd.Series:
        """
        Calculate match scores based on role requirements.
        
        Args:
            required_roles: Dictionary with roles as keys and counts as values.
            
        Returns:
            Series with resource IDs as index and role match scores as values.
        """
        # Initialize scores
        role_scores = pd.Series(0, index=self.resource_metadata.index)
        
        # Check if 'role' column exists in metadata
        if 'role' not in self.resource_metadata.columns:
            return role_scores
            
        # Score resources based on required roles
        for role, count in required_roles.items():
            # Find resources with matching roles
            matching_resources = self.resource_metadata[self.resource_metadata['role'] == role].index
            
            # Assign score based on the importance (count) of the role
            role_scores.loc[matching_resources] += count
            
        # Normalize scores
        if role_scores.max() > 0:
            role_scores = role_scores / role_scores.max()
            
        return role_scores
        
    def _combine_scores(self, skill_scores: pd.Series, role_scores: Optional[pd.Series] = None,
                       skill_weight: float = 0.7, role_weight: float = 0.3) -> pd.Series:
        """
        Combine different score components.
        
        Args:
            skill_scores: Series with skill match scores.
            role_scores: Optional series with role match scores.
            skill_weight: Weight for skill scores.
            role_weight: Weight for role scores.
            
        Returns:
            Series with combined scores.
        """
        # If no role scores, use only skill scores
        if role_scores is None:
            return skill_scores
            
        # Ensure indexes match
        common_index = skill_scores.index.intersection(role_scores.index)
        
        # Combine scores with weights
        combined_scores = pd.Series(
            skill_weight * skill_scores.loc[common_index] + 
            role_weight * role_scores.loc[common_index],
            index=common_index
        )
        
        return combined_scores