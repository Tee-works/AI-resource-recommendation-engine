# scripts/generate_data.py
import sys
import os

# Add the src directory to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_generator import DataGenerator

def main():
    """Generate synthetic data for the resource recommendation engine."""
    print("Generating synthetic data for Float Resource Recommendation Engine...")
    
    # Initialize data generator
    generator = DataGenerator(seed=42)
    
    # Generate data
    data = generator.generate_data(
        n_projects=50,
        n_resources=30,
        start_date="2022-01-01",
        end_date="2023-12-31"
    )
    
    # Save data
    generator.save_data(data, output_dir="data/raw")
    
    print("Data generation complete!")
    print("You can now explore the data in notebooks/01_data_exploration.ipynb")

if __name__ == "__main__":
    main()