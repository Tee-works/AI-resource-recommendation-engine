# src/data/data_generator.py
import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

class DataGenerator:
    """Generate synthetic data for the resource recommendation engine."""
    
    def __init__(self, seed=42):
        """Initialize the data generator with a random seed for reproducibility."""
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        # Define common attributes
        self.roles = ["Developer", "Designer", "Project Manager", "QA Engineer", "DevOps Engineer"]
        self.skills = ["Python", "JavaScript", "UI Design", "UX Research", "Project Management", 
                       "Agile", "DevOps", "Testing", "React", "Node.js", "Data Analysis", 
                       "Cloud Services", "Mobile Development", "Security", "Database"]
        self.departments = ["Engineering", "Design", "Management", "QA", "Operations"]
        
    def generate_data(self, n_projects=50, n_resources=30, start_date="2022-01-01", end_date="2023-12-31"):
        """Generate a complete dataset with projects, resources, skills, and allocations."""
        # Generate individual datasets
        projects = self._generate_projects(n_projects, start_date, end_date)
        resources = self._generate_resources(n_resources)
        skills = self._generate_resource_skills(resources)
        allocations = self._generate_allocations(projects, resources)
        feedback = self._generate_project_feedback(projects, resources, allocations)
        
        # Return as dictionary
        return {
            "projects": projects,
            "resources": resources,
            "resource_skills": skills,
            "allocations": allocations,
            "feedback": feedback
        }
        
    def _generate_projects(self, n_projects, start_date, end_date):
        """Generate project data."""
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        date_range = (end - start).days
        
        data = []
        for i in range(1, n_projects + 1):
            project_start = start + timedelta(days=random.randint(0, date_range - 30))
            project_length = random.randint(7, 90)  # Project length between 1 week and 3 months
            project_end = project_start + timedelta(days=project_length)
            
            # Randomize success metrics
            is_successful = random.random() > 0.3  # 70% success rate
            
            success_score = random.uniform(0.7, 1.0) if is_successful else random.uniform(0.3, 0.7)
            on_time = random.random() > 0.2 if is_successful else random.random() > 0.7
            on_budget = random.random() > 0.2 if is_successful else random.random() > 0.7
            
            client_satisfaction = random.uniform(4.0, 5.0) if is_successful else random.uniform(2.0, 4.0)
            team_satisfaction = random.uniform(3.5, 5.0) if is_successful else random.uniform(2.0, 3.5)
            
            data.append({
                "id": f"p{i}",
                "name": f"Project {i}",
                "client_id": f"c{random.randint(1, 10)}",
                "start_date": project_start,
                "end_date": project_end,
                "budget": random.randint(10000, 100000),
                "status": "completed" if project_end < end else "active",
                "success_score": success_score,
                "on_time": on_time,
                "on_budget": on_budget,
                "client_satisfaction": client_satisfaction,
                "team_satisfaction": team_satisfaction
            })
            
        return pd.DataFrame(data)
        
    def _generate_resources(self, n_resources):
        """Generate resource (team member) data."""
        data = []
        for i in range(1, n_resources + 1):
            role = random.choice(self.roles)
            department = self._get_department_for_role(role)
            
            # Simulate different experience levels
            experience_level = random.choice(["Junior", "Mid", "Senior"])
            hourly_rate = self._get_hourly_rate(role, experience_level)
            
            data.append({
                "id": f"r{i}",
                "name": f"Resource {i}",
                "email": f"resource{i}@example.com",
                "role": role,
                "department": department,
                "experience_level": experience_level,
                "hourly_rate": hourly_rate,
                "capacity": 40,  # Standard 40-hour work week
                "join_date": pd.to_datetime("2021-01-01") + timedelta(days=random.randint(0, 365))
            })
            
        return pd.DataFrame(data)
        
    def _generate_resource_skills(self, resources):
        """Generate skills for each resource."""
        data = []
        
        for _, resource in resources.iterrows():
            # Determine number of skills based on experience
            if resource["experience_level"] == "Junior":
                n_skills = random.randint(2, 4)
            elif resource["experience_level"] == "Mid":
                n_skills = random.randint(4, 7)
            else:  # Senior
                n_skills = random.randint(6, 10)
                
            # Get role-appropriate skills
            available_skills = self._get_skills_for_role(resource["role"])
            
            # If not enough role-specific skills, add some generic ones
            if len(available_skills) < n_skills:
                additional_skills = [s for s in self.skills if s not in available_skills]
                available_skills.extend(additional_skills)
                
            # Select random skills
            selected_skills = random.sample(available_skills, min(n_skills, len(available_skills)))
            
            # Add skills with proficiency levels
            for skill in selected_skills:
                # Higher experience = higher proficiency on average
                if resource["experience_level"] == "Junior":
                    proficiency = random.randint(1, 3)
                elif resource["experience_level"] == "Mid":
                    proficiency = random.randint(2, 4)
                else:  # Senior
                    proficiency = random.randint(3, 5)
                    
                data.append({
                    "resource_id": resource["id"],
                    "skill_name": skill,
                    "proficiency_level": proficiency
                })
                
        return pd.DataFrame(data)
        
    def _generate_allocations(self, projects, resources):
        """Generate resource allocations to projects."""
        data = []
        allocation_id = 1
        
        for _, project in projects.iterrows():
            # Determine team size based on project budget and duration
            project_duration = (project["end_date"] - project["start_date"]).days
            team_budget_factor = project["budget"] / 50000  # Normalize budget
            team_size = max(2, int(team_budget_factor * 5))  # At least 2 people, scale with budget
            
            # Ensure we don't try to allocate more resources than available
            team_size = min(team_size, len(resources))
            
            # Select random resources
            team_resources = resources.sample(team_size)
            
            # Create allocations
            for _, resource in team_resources.iterrows():
                # Some resources might not be allocated for the full project duration
                allocation_start = project["start_date"]
                allocation_end = project["end_date"]
                
                # Part-time or full-time
                hours_per_day = random.choice([4, 6, 8])
                
                data.append({
                    "id": f"a{allocation_id}",
                    "project_id": project["id"],
                    "resource_id": resource["id"],
                    "start_date": allocation_start,
                    "end_date": allocation_end,
                    "hours_per_day": hours_per_day,
                    "role": resource["role"]
                })
                
                allocation_id += 1
                
        return pd.DataFrame(data)
        
    def _generate_project_feedback(self, projects, resources, allocations):
        """Generate feedback for completed projects."""
        data = []
        feedback_id = 1
        
        # Get completed projects
        completed_projects = projects[projects["status"] == "completed"]
        
        for _, project in completed_projects.iterrows():
            # Get resources allocated to this project
            project_allocations = allocations[allocations["project_id"] == project["id"]]
            project_resources = project_allocations["resource_id"].unique()
            
            # Generate feedback for some resources (not all might receive feedback)
            for resource_id in project_resources:
                if random.random() < 0.8:  # 80% chance of receiving feedback
                    # Rating correlates with project success
                    base_rating = 3
                    if project["success_score"] > 0.7:
                        base_rating = 4
                    
                    # Add some randomness
                    rating = min(5, max(1, base_rating + random.randint(-1, 1)))
                    
                    data.append({
                        "id": f"f{feedback_id}",
                        "project_id": project["id"],
                        "resource_id": resource_id,
                        "submitted_by": project["client_id"],
                        "rating": rating,
                        "feedback_text": self._generate_feedback_text(rating),
                        "created_at": project["end_date"] + timedelta(days=random.randint(1, 7))
                    })
                    
                    feedback_id += 1
                    
        return pd.DataFrame(data)
        
    def _get_department_for_role(self, role):
        """Map role to appropriate department."""
        role_dept_map = {
            "Developer": "Engineering",
            "Designer": "Design",
            "Project Manager": "Management",
            "QA Engineer": "QA",
            "DevOps Engineer": "Operations"
        }
        return role_dept_map.get(role, random.choice(self.departments))
        
    def _get_hourly_rate(self, role, experience_level):
        """Determine hourly rate based on role and experience."""
        base_rates = {
            "Developer": 80,
            "Designer": 75,
            "Project Manager": 90,
            "QA Engineer": 70,
            "DevOps Engineer": 85
        }
        
        experience_multipliers = {
            "Junior": 0.7,
            "Mid": 1.0,
            "Senior": 1.3
        }
        
        base_rate = base_rates.get(role, 75)
        multiplier = experience_multipliers.get(experience_level, 1.0)
        
        # Add some randomness
        return int(base_rate * multiplier * random.uniform(0.9, 1.1))
        
    def _get_skills_for_role(self, role):
        """Get skills that are appropriate for a given role."""
        role_skills_map = {
            "Developer": ["Python", "JavaScript", "React", "Node.js", "Database", "Mobile Development"],
            "Designer": ["UI Design", "UX Research", "React"],
            "Project Manager": ["Project Management", "Agile", "Data Analysis"],
            "QA Engineer": ["Testing", "Python", "Security"],
            "DevOps Engineer": ["DevOps", "Cloud Services", "Security", "Python"]
        }
        return role_skills_map.get(role, random.sample(self.skills, 3))
        
    def _generate_feedback_text(self, rating):
        """Generate feedback text based on rating."""
        positive_feedback = [
            "Great work on this project!",
            "Excellent contribution to the team.",
            "Very professional and skilled resource.",
            "Delivered high-quality work consistently.",
            "A pleasure to work with, highly recommended."
        ]
        
        neutral_feedback = [
            "Completed the work as expected.",
            "Met the requirements adequately.",
            "Solid performance overall.",
            "Worked well with the team.",
            "Delivered on time with acceptable quality."
        ]
        
        negative_feedback = [
            "Struggled to meet expectations.",
            "Work required significant revisions.",
            "Communication issues affected delivery.",
            "Missed several deadlines.",
            "Quality of work was below expectations."
        ]
        
        if rating >= 4:
            return random.choice(positive_feedback)
        elif rating >= 3:
            return random.choice(neutral_feedback)
        else:
            return random.choice(negative_feedback)
        
    def save_data(self, data, output_dir="data/raw"):
        """Save generated data to CSV files."""
        import os
        
        # Create directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save each dataset to a CSV file
        for name, dataset in data.items():
            filename = f"{output_dir}/{name}.csv"
            dataset.to_csv(filename, index=False)
            print(f"Saved {len(dataset)} records to {filename}")