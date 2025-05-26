# src/models/constraint_optimizer.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import pulp

class ConstraintOptimizer:
    """Constraint optimization model for team composition."""
    
    def __init__(self):
        """Initialize the constraint optimizer."""
        self.resource_metadata = None
        self.resource_skill_matrix = None
        self.is_trained = False
        
    def train(self, resource_metadata: pd.DataFrame, resource_skill_matrix: Optional[pd.DataFrame] = None):
        """
        Train the constraint optimizer with resource data.
        
        Args:
            resource_metadata: DataFrame with resource metadata (hourly rate, role, etc.)
            resource_skill_matrix: Optional DataFrame with resource skill proficiencies
        """
        self.resource_metadata = resource_metadata
        self.resource_skill_matrix = resource_skill_matrix
        self.is_trained = True
        
    def optimize_team(self, initial_scores: pd.Series, project_requirements: Dict, 
                     team_size: int = 5) -> pd.Series:
        """
        Optimize team composition based on initial scores and constraints.
        
        Args:
            initial_scores: Series with resource IDs as index and initial recommendation scores as values.
            project_requirements: Dictionary with project requirements including constraints.
            team_size: Target team size.
            
        Returns:
            Series with optimized resource recommendations.
        """
        if not self.is_trained:
            raise ValueError("Optimizer has not been trained. Call train() first.")
            
        # Create optimization model
        model = pulp.LpProblem(name="team_optimization", sense=pulp.LpMaximize)
        
        # Create decision variables (1 if resource is selected, 0 otherwise)
        resource_vars = {}
        for resource_id in initial_scores.index:
            resource_vars[resource_id] = pulp.LpVariable(f"resource_{resource_id}", cat='Binary')
            
        # Objective function: maximize sum of resource scores
        model += pulp.lpSum([resource_vars[resource_id] * score for resource_id, score in initial_scores.items()])
        
        # Constraint: team size
        model += pulp.lpSum(resource_vars.values()) == team_size, "team_size_constraint"
        
        # Add role constraints if specified
        if 'required_roles' in project_requirements and self.resource_metadata is not None and 'role' in self.resource_metadata.columns:
            required_roles = project_requirements['required_roles']
            
            for role, count in required_roles.items():
                # Find resources with this role
                resources_with_role = self.resource_metadata[self.resource_metadata['role'] == role].index
                
                # Add constraint that at least 'count' resources with this role must be selected
                model += pulp.lpSum([resource_vars[r] for r in resources_with_role if r in resource_vars]) >= count, f"role_{role}_constraint"
                
        # Add budget constraint if specified
        if 'budget' in project_requirements and self.resource_metadata is not None and 'hourly_rate' in self.resource_metadata.columns:
            budget = project_requirements['budget']
            
            # Estimate project duration based on requirements or use default
            project_duration = project_requirements.get('duration_days', 30) # Default to 30 days
            hours_per_day = project_requirements.get('hours_per_day', 8) # Default to 8 hours/day
            
            # Calculate total cost for each resource
            resource_costs = {}
            for resource_id in resource_vars:
                if resource_id in self.resource_metadata.index:
                    hourly_rate = self.resource_metadata.loc[resource_id, 'hourly_rate']
                    resource_costs[resource_id] = hourly_rate * hours_per_day * project_duration
                else:
                    # Use average rate if resource not found
                    avg_rate = self.resource_metadata['hourly_rate'].mean()
                    resource_costs[resource_id] = avg_rate * hours_per_day * project_duration
                    
            # Add budget constraint
            model += pulp.lpSum([resource_vars[r] * resource_costs[r] for r in resource_vars]) <= budget, "budget_constraint"
            
        # Add skill coverage constraints if specified
        if 'required_skills' in project_requirements and self.resource_skill_matrix is not None:
            required_skills = project_requirements['required_skills']
            
            # For each required skill, ensure at least one team member has the skill
            for skill, importance in required_skills.items():
                if skill in self.resource_skill_matrix.columns:
                    # Get resources with this skill (non-zero proficiency)
                    resources_with_skill = self.resource_skill_matrix[self.resource_skill_matrix[skill] > 0].index
                    
                    # Only add constraint if there are resources with this skill
                    if len(resources_with_skill) > 0:
                        # Add constraint that at least one resource with this skill must be selected
                        model += pulp.lpSum([resource_vars[r] for r in resources_with_skill if r in resource_vars]) >= 1, f"skill_{skill}_constraint"
                        
        # Add diversity constraint if specified
        if 'diversity' in project_requirements and self.resource_metadata is not None and 'department' in self.resource_metadata.columns:
            min_departments = project_requirements['diversity'].get('min_departments', 2)
            
            # Get unique departments
            departments = self.resource_metadata['department'].unique()
            
            # Create variables to track if at least one resource from each department is selected
            dept_vars = {}
            for dept in departments:
                dept_vars[dept] = pulp.LpVariable(f"dept_{dept}", cat='Binary')
                
                # Get resources in this department
                resources_in_dept = self.resource_metadata[self.resource_metadata['department'] == dept].index
                resources_in_dept = [r for r in resources_in_dept if r in resource_vars]
                
                if resources_in_dept:
                    # dept_var = 1 if at least one resource from this department is selected
                    for r in resources_in_dept:
                        model += dept_vars[dept] >= resource_vars[r], f"dept_{dept}_resource_{r}_implication"
                        
                    # If no resources from this department are selected, dept_var must be 0
                    model += dept_vars[dept] <= pulp.lpSum([resource_vars[r] for r in resources_in_dept]), f"dept_{dept}_zero_implication"
                    
            # Ensure at least min_departments are represented
            model += pulp.lpSum(dept_vars.values()) >= min_departments, "min_departments_constraint"
            
        # Solve the problem
        model.solve(pulp.PULP_CBC_CMD(msg=False))
        
        # Check if a solution was found
        if model.status != pulp.LpStatusOptimal:
            # If no optimal solution, return the initial scores
            return initial_scores.sort_values(ascending=False).head(team_size)
            
        # Get the optimal team
        optimized_scores = pd.Series(dtype=float)
        
        for resource_id, var in resource_vars.items():
            if var.value() == 1:  # Resource is selected in the optimal team
                optimized_scores[resource_id] = initial_scores[resource_id]
                
        return optimized_scores
        
    def get_team_cost(self, team: List[str], project_requirements: Dict) -> float:
        """
        Calculate the estimated cost of a team for this project.
        
        Args:
            team: List of resource IDs in the team.
            project_requirements: Dictionary with project requirements.
            
        Returns:
            Estimated total cost.
        """
        if self.resource_metadata is None or 'hourly_rate' not in self.resource_metadata.columns:
            return 0.0
            
        # Estimate project duration based on requirements or use default
        project_duration = project_requirements.get('duration_days', 30) # Default to 30 days
        hours_per_day = project_requirements.get('hours_per_day', 8) # Default to 8 hours/day
        
        # Calculate total cost
        total_cost = 0.0
        
        for resource_id in team:
            if resource_id in self.resource_metadata.index:
                hourly_rate = self.resource_metadata.loc[resource_id, 'hourly_rate']
                resource_cost = hourly_rate * hours_per_day * project_duration
                total_cost += resource_cost
                
        return total_cost
        
    def get_skill_coverage(self, team: List[str], required_skills: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate how well the team covers the required skills.
        
        Args:
            team: List of resource IDs in the team.
            required_skills: Dictionary mapping skill names to importance values.
            
        Returns:
            Dictionary with skill coverage metrics.
        """
        if self.resource_skill_matrix is None:
            return {'coverage_percentage': 0.0, 'skills_covered': 0, 'total_skills': len(required_skills)}
            
        # Count covered skills
        covered_skills = 0
        skill_coverage = {}
        
        for skill, importance in required_skills.items():
            if skill in self.resource_skill_matrix.columns:
                # Check if any team member has this skill
                team_skill_levels = [
                    self.resource_skill_matrix.loc[r, skill] 
                    for r in team 
                    if r in self.resource_skill_matrix.index
                ]
                
                if team_skill_levels and max(team_skill_levels) > 0:
                    covered_skills += 1
                    skill_coverage[skill] = max(team_skill_levels)
                else:
                    skill_coverage[skill] = 0.0
            else:
                skill_coverage[skill] = 0.0
                
        # Calculate coverage percentage
        coverage_percentage = (covered_skills / len(required_skills)) * 100 if required_skills else 100.0
        
        return {
            'coverage_percentage': coverage_percentage,
            'skills_covered': covered_skills,
            'total_skills': len(required_skills),
            'skill_coverage': skill_coverage
        }