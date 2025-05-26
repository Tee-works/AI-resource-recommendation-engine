# src/utils/evaluation.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_recommendations(model, test_data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Evaluate recommendation model using test data.
    
    Args:
        model: Trained recommendation model.
        test_data: Dictionary containing test datasets.
        
    Returns:
        Dictionary with evaluation metrics.
    """
    metrics = {}
    
    # Get test projects with known outcomes
    if 'projects' in test_data and 'success_score' in test_data['projects'].columns:
        test_projects = test_data['projects']
        
        # Get allocations for test projects
        if 'allocations' in test_data:
            test_allocations = test_data['allocations']
            
            # Calculate metrics for each test project
            project_metrics = []
            
            for _, project in test_projects.iterrows():
                project_id = project['id']
                
                # Get actual team for this project
                actual_team = test_allocations[test_allocations['project_id'] == project_id]['resource_id'].unique().tolist()
                
                if not actual_team:
                    continue
                    
                # Create project requirements (simplified for evaluation)
                # In a real system, we'd reconstruct more detailed requirements
                project_requirements = {
                    'id': project_id,
                    'name': project.get('name', f'Project {project_id}'),
                    'required_roles': {}  # Simplified for evaluation
                }
                
                # Get recommended team
                try:
                    # Exclude actual team to simulate cold start recommendation
                    recommendations = model.recommend_team(
                        project_requirements,
                        team_size=len(actual_team),
                        existing_team=[]
                    )
                    
                    recommended_team = recommendations.index.tolist()
                    
                    # Calculate team overlap (precision)
                    overlap = len(set(recommended_team) & set(actual_team))
                    precision = overlap / len(recommended_team) if recommended_team else 0
                    recall = overlap / len(actual_team) if actual_team else 0
                    
                    # Calculate F1 score
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                    
                    project_metrics.append({
                        'project_id': project_id,
                        'success_score': project['success_score'],
                        'team_size': len(actual_team),
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    })
                    
                except Exception as e:
                    print(f"Error evaluating project {project_id}: {e}")
                    continue
            
            # Calculate aggregate metrics
            if project_metrics:
                metrics_df = pd.DataFrame(project_metrics)
                
                # Overall metrics
                metrics['avg_precision'] = metrics_df['precision'].mean()
                metrics['avg_recall'] = metrics_df['recall'].mean()
                metrics['avg_f1_score'] = metrics_df['f1_score'].mean()
                
                # Correlation between recommendation quality and project success
                success_correlation = metrics_df[['f1_score', 'success_score']].corr().iloc[0, 1]
                metrics['success_correlation'] = success_correlation
                
    return metrics

def evaluate_skill_coverage(recommended_team: List[str], 
                           required_skills: Dict[str, float],
                           resource_skills: pd.DataFrame) -> Dict[str, float]:
    """
    Evaluate how well the recommended team covers the required skills.
    
    Args:
        recommended_team: List of resource IDs in the recommended team.
        required_skills: Dictionary mapping skill names to importance values.
        resource_skills: DataFrame with resource skills data.
        
    Returns:
        Dictionary with skill coverage metrics.
    """
    # Filter skills for recommended team members
    team_skills = resource_skills[resource_skills['resource_id'].isin(recommended_team)]
    
    # Calculate coverage metrics
    total_skills = len(required_skills)
    covered_skills = 0
    coverage_score = 0.0
    
    for skill, importance in required_skills.items():
        # Check if any team member has this skill
        skill_coverage = team_skills[team_skills['skill_name'] == skill]
        
        if not skill_coverage.empty:
            covered_skills += 1
            
            # Calculate weighted coverage score based on proficiency
            max_proficiency = skill_coverage['proficiency_level'].max()
            normalized_proficiency = max_proficiency / 5.0  # Assuming 5 is max proficiency
            coverage_score += importance * normalized_proficiency
            
    # Normalize coverage score
    if total_skills > 0:
        coverage_percentage = (covered_skills / total_skills) * 100
        normalized_coverage_score = coverage_score / sum(required_skills.values())
    else:
        coverage_percentage = 100.0
        normalized_coverage_score = 1.0
        
    return {
        'coverage_percentage': coverage_percentage,
        'normalized_coverage_score': normalized_coverage_score,
        'covered_skills': covered_skills,
        'total_skills': total_skills
    }