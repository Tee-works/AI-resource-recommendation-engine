# src/visualization/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

def plot_team_recommendation(recommended_team: pd.Series, 
                            resource_metadata: Optional[pd.DataFrame] = None,
                            resource_skills: Optional[pd.DataFrame] = None,
                            save_path: Optional[str] = None):
    """
    Plot team recommendation results.
    
    Args:
        recommended_team: Series with resource IDs as index and recommendation scores as values.
        resource_metadata: Optional DataFrame with resource metadata.
        resource_skills: Optional DataFrame with resource skills.
        save_path: Optional path to save the plot.
    """
    # Set up the figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Sort recommendations by score
    sorted_team = recommended_team.sort_values(ascending=False)
    
    # Plot recommendation scores
    ax1 = axes[0]
    sorted_team.plot(kind='barh', ax=ax1)
    ax1.set_title('Recommended Team Members by Score')
    ax1.set_xlabel('Recommendation Score')
    ax1.set_ylabel('Resource ID')
    
    # Add role information if available
    if resource_metadata is not None and 'role' in resource_metadata.columns:
        # Extract roles for recommended team
        team_roles = []
        for resource_id in sorted_team.index:
            if resource_id in resource_metadata.index:
                role = resource_metadata.loc[resource_id, 'role']
                team_roles.append(role)
            else:
                team_roles.append('Unknown')
                
        # Add role labels to the plot
        for i, (resource_id, score) in enumerate(sorted_team.items()):
            role = team_roles[i] if i < len(team_roles) else 'Unknown'
            ax1.text(score + 0.01, i, role, va='center')
            
    # Plot skill distribution if available
    ax2 = axes[1]
    if resource_skills is not None:
        # Get skills for recommended team
        team_skills = resource_skills[resource_skills['resource_id'].isin(sorted_team.index)]
        
        # Count skills
        skill_counts = team_skills['skill_name'].value_counts()
        
        # Plot top 10 skills
        top_skills = skill_counts.head(10)
        sns.barplot(x=top_skills.values, y=top_skills.index, ax=ax2)
        ax2.set_title('Top Skills in Recommended Team')
        ax2.set_xlabel('Count')
        ax2.set_ylabel('Skill')
    else:
        ax2.set_title('Skill data not available')
        
    # Adjust layout
    plt.tight_layout()
    
    # Save plot if requested
    if save_path:
        plt.savefig(save_path)
        
    # Show plot
    plt.show()

def plot_project_success_factors(projects: pd.DataFrame, allocations: pd.DataFrame):
    """
    Plot factors influencing project success.
    
    Args:
        projects: DataFrame with project data.
        allocations: DataFrame with allocation data.
    """
    # Set up the figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot success score distribution
    ax1 = axes[0, 0]
    sns.histplot(projects['success_score'], bins=20, kde=True, ax=ax1)
    ax1.set_title('Distribution of Project Success Scores')
    ax1.set_xlabel('Success Score')
    ax1.set_ylabel('Count')
    
    # Calculate team size for each project
    team_sizes = allocations.groupby('project_id')['resource_id'].nunique()
    project_team_sizes = projects.merge(
        team_sizes.reset_index().rename(columns={'resource_id': 'team_size'}),
        left_on='id',
        right_on='project_id',
        how='left'
    )
    
    # Plot team size vs. success score
    ax2 = axes[0, 1]
    sns.scatterplot(x='team_size', y='success_score', data=project_team_sizes, ax=ax2)
    ax2.set_title('Team Size vs. Success Score')
    ax2.set_xlabel('Team Size')
    ax2.set_ylabel('Success Score')
    
    # Plot success metrics correlation
    ax3 = axes[1, 0]
    success_metrics = ['success_score', 'on_time', 'on_budget', 'client_satisfaction', 'team_satisfaction']
    success_metrics = [col for col in success_metrics if col in projects.columns]
    
    if len(success_metrics) > 1:
        sns.heatmap(projects[success_metrics].corr(), annot=True, cmap='coolwarm', ax=ax3)
        ax3.set_title('Correlation Between Success Metrics')
    else:
        ax3.set_title('Insufficient success metrics for correlation analysis')
        
    # Plot success over time
    ax4 = axes[1, 1]
    if 'end_date' in projects.columns:
        projects['end_date'] = pd.to_datetime(projects['end_date'])
        projects_sorted = projects.sort_values('end_date')
        
        sns.lineplot(x='end_date', y='success_score', data=projects_sorted, ax=ax4)
        ax4.set_title('Project Success Over Time')
        ax4.set_xlabel('Project End Date')
        ax4.set_ylabel('Success Score')
    else:
        ax4.set_title('End date not available for time analysis')
        
    # Adjust layout
    plt.tight_layout()
    
    # Show plot
    plt.show()

def plot_skill_importance(model, resource_skills: pd.DataFrame):
    """
    Plot skill importance based on model weights.
    
    Args:
        model: Trained recommendation model.
        resource_skills: DataFrame with resource skills.
    """
    # This is a placeholder implementation
    # In a real system, we would extract actual feature importances from the model
    
    # Plot top skills by frequency
    plt.figure(figsize=(12, 6))
    
    skill_counts = resource_skills['skill_name'].value_counts().head(15)
    sns.barplot(x=skill_counts.values, y=skill_counts.index)
    
    plt.title('Most Common Skills (Proxy for Importance)')
    plt.xlabel('Count')
    plt.ylabel('Skill')
    
    plt.tight_layout()
    plt.show()