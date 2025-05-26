# src/data/data_processor.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import os

class DataProcessor:
    """Process raw data for the recommendation engine."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """Initialize data processor."""
        self.data_dir = data_dir
        self.processed_data_dir = "data/processed"
        self.raw_data = {}
        self.processed_data = {}
        
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load raw data from CSV files."""
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Data directory {self.data_dir} not found.")
            
        # Load each CSV file in the data directory
        for filename in os.listdir(self.data_dir):
            if filename.endswith(".csv"):
                file_path = os.path.join(self.data_dir, filename)
                dataset_name = filename.split(".")[0]
                
                # Skip empty or hidden files
                if os.path.getsize(file_path) == 0 or filename.startswith("."):
                    continue
                    
                # Load data
                print(f"Loading {file_path}...")
                self.raw_data[dataset_name] = pd.read_csv(file_path)
                
        return self.raw_data
        
    def process_data(self) -> Dict[str, pd.DataFrame]:
        """Process raw data for modeling."""
        if not self.raw_data:
            self.load_data()
            
        # Process projects data
        if 'projects' in self.raw_data:
            projects = self.raw_data['projects'].copy()
            
            # Convert date columns to datetime
            for date_col in ['start_date', 'end_date']:
                if date_col in projects.columns:
                    projects[date_col] = pd.to_datetime(projects[date_col])
            
            # Create project duration feature
            if 'start_date' in projects.columns and 'end_date' in projects.columns:
                projects['duration_days'] = (projects['end_date'] - projects['start_date']).dt.days
                
            self.processed_data['projects'] = projects
            
        # Process resources data
        if 'resources' in self.raw_data:
            resources = self.raw_data['resources'].copy()
            
            # Convert join_date to datetime if it exists
            if 'join_date' in resources.columns:
                resources['join_date'] = pd.to_datetime(resources['join_date'])
                
            # Create experience features if join_date exists
            if 'join_date' in resources.columns:
                latest_date = pd.to_datetime('today')
                resources['experience_days'] = (latest_date - resources['join_date']).dt.days
                
            self.processed_data['resources'] = resources
            
        # Process resource skills data
        if 'resource_skills' in self.raw_data:
            resource_skills = self.raw_data['resource_skills'].copy()
            
            # Normalize proficiency levels
            if 'proficiency_level' in resource_skills.columns:
                max_proficiency = resource_skills['proficiency_level'].max()
                resource_skills['normalized_proficiency'] = resource_skills['proficiency_level'] / max_proficiency
                
            self.processed_data['resource_skills'] = resource_skills
            
        # Process allocations data
        if 'allocations' in self.raw_data:
            allocations = self.raw_data['allocations'].copy()
            
            # Convert date columns to datetime
            for date_col in ['start_date', 'end_date']:
                if date_col in allocations.columns:
                    allocations[date_col] = pd.to_datetime(allocations[date_col])
                    
            # Create allocation duration feature
            if 'start_date' in allocations.columns and 'end_date' in allocations.columns:
                allocations['allocation_days'] = (allocations['end_date'] - allocations['start_date']).dt.days
                
            # Calculate total hours
            if 'allocation_days' in allocations.columns and 'hours_per_day' in allocations.columns:
                allocations['total_hours'] = allocations['allocation_days'] * allocations['hours_per_day']
                
            self.processed_data['allocations'] = allocations
            
        # Process feedback data
        if 'feedback' in self.raw_data:
            feedback = self.raw_data['feedback'].copy()
            
            # Convert date columns to datetime
            if 'created_at' in feedback.columns:
                feedback['created_at'] = pd.to_datetime(feedback['created_at'])
                
            self.processed_data['feedback'] = feedback
            
        return self.processed_data
        
    def create_feature_matrices(self) -> Dict[str, pd.DataFrame]:
        """Create feature matrices for recommendation models."""
        if not self.processed_data:
            self.process_data()
            
        feature_matrices = {}
        
        # Create resource-skill matrix
        if 'resource_skills' in self.processed_data and 'resources' in self.processed_data:
            # Create pivot table of resources and their skills
            resource_skills = self.processed_data['resource_skills']
            
            if 'normalized_proficiency' in resource_skills.columns:
                proficiency_col = 'normalized_proficiency'
            else:
                proficiency_col = 'proficiency_level'
                
            # Create pivot table
            resource_skill_matrix = resource_skills.pivot_table(
                index='resource_id',
                columns='skill_name',
                values=proficiency_col,
                fill_value=0
            )
            
            feature_matrices['resource_skill_matrix'] = resource_skill_matrix
            
        # Create project-resource matrix
        if 'allocations' in self.processed_data:
            allocations = self.processed_data['allocations']
            
            # Create a matrix showing which resources worked on which projects
            if 'total_hours' in allocations.columns:
                values_col = 'total_hours'
            else:
                values_col = 'hours_per_day'
                
            project_resource_matrix = allocations.pivot_table(
                index='project_id',
                columns='resource_id',
                values=values_col,
                aggfunc='sum',
                fill_value=0
            )
            
            feature_matrices['project_resource_matrix'] = project_resource_matrix
            
        # Create project success vector
        if 'projects' in self.processed_data:
            projects = self.processed_data['projects']
            
            if 'success_score' in projects.columns:
                project_success = projects[['id', 'success_score']].set_index('id')
                feature_matrices['project_success'] = project_success
                
        # Create resource metadata matrix
        if 'resources' in self.processed_data:
            resources = self.processed_data['resources']
            
            # Select relevant columns for metadata
            metadata_cols = ['id', 'role', 'department', 'experience_level', 'hourly_rate']
            metadata_cols = [col for col in metadata_cols if col in resources.columns]
            
            if metadata_cols:
                resource_metadata = resources[metadata_cols].set_index('id')
                feature_matrices['resource_metadata'] = resource_metadata
                
        return feature_matrices
        
    def save_processed_data(self):
        """Save processed data to CSV files."""
        if not self.processed_data:
            self.process_data()
            
        # Create processed data directory if it doesn't exist
        os.makedirs(self.processed_data_dir, exist_ok=True)
        
        # Save each processed dataset to a CSV file
        for name, dataset in self.processed_data.items():
            file_path = os.path.join(self.processed_data_dir, f"{name}.csv")
            dataset.to_csv(file_path, index=False)
            print(f"Saved processed {name} data to {file_path}")
            
    def create_train_test_split(self, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Dict[str, pd.DataFrame]]:
        """Split data into training and test sets."""
        from sklearn.model_selection import train_test_split
        
        if not self.processed_data:
            self.process_data()
            
        splits = {}
        
        # If we have projects data, use it to create a consistent split across related datasets
        if 'projects' in self.processed_data:
            projects = self.processed_data['projects']
            
            # Get project IDs
            project_ids = projects['id'].unique()
            
            # Split project IDs into train and test sets
            train_project_ids, test_project_ids = train_test_split(
                project_ids, test_size=test_size, random_state=random_state
            )
            
            # Split projects data
            train_projects = projects[projects['id'].isin(train_project_ids)]
            test_projects = projects[projects['id'].isin(test_project_ids)]
            
            splits['projects'] = {
                'train': train_projects,
                'test': test_projects
            }
            
            # Split allocations data based on project IDs
            if 'allocations' in self.processed_data:
                allocations = self.processed_data['allocations']
                
                train_allocations = allocations[allocations['project_id'].isin(train_project_ids)]
                test_allocations = allocations[allocations['project_id'].isin(test_project_ids)]
                
                splits['allocations'] = {
                    'train': train_allocations,
                    'test': test_allocations
                }
                
            # Split feedback data based on project IDs
            if 'feedback' in self.processed_data:
                feedback = self.processed_data['feedback']
                
                train_feedback = feedback[feedback['project_id'].isin(train_project_ids)]
                test_feedback = feedback[feedback['project_id'].isin(test_project_ids)]
                
                splits['feedback'] = {
                    'train': train_feedback,
                    'test': test_feedback
                }
                
        # Split resources and resource skills data randomly
        if 'resources' in self.processed_data:
            resources = self.processed_data['resources']
            
            train_resources, test_resources = train_test_split(
                resources, test_size=test_size, random_state=random_state
            )
            
            splits['resources'] = {
                'train': train_resources,
                'test': test_resources
            }
            
        if 'resource_skills' in self.processed_data:
            resource_skills = self.processed_data['resource_skills']
            
            # If we have resource splits, use them for consistency
            if 'resources' in splits:
                train_resource_ids = splits['resources']['train']['id'].unique()
                test_resource_ids = splits['resources']['test']['id'].unique()
                
                train_resource_skills = resource_skills[resource_skills['resource_id'].isin(train_resource_ids)]
                test_resource_skills = resource_skills[resource_skills['resource_id'].isin(test_resource_ids)]
            else:
                # Otherwise split randomly
                train_resource_skills, test_resource_skills = train_test_split(
                    resource_skills, test_size=test_size, random_state=random_state
                )
                
            splits['resource_skills'] = {
                'train': train_resource_skills,
                'test': test_resource_skills
            }
            
        return splits