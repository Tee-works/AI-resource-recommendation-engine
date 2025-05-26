document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const projectForm = document.getElementById('project-form');
    const projectName = document.getElementById('project-name');
    const teamSize = document.getElementById('team-size');
    const teamSizeValue = document.getElementById('team-size-value');
    const skillsContainer = document.getElementById('skills-container');
    const addSkillBtn = document.getElementById('add-skill');
    const rolesContainer = document.getElementById('roles-container');
    const addRoleBtn = document.getElementById('add-role');
    const submitButton = document.getElementById('submit-button');
    const recommendationContainer = document.getElementById('recommendation-container');
    const recommendationProjectName = document.getElementById('recommendation-project-name');
    const recommendationTeamSize = document.getElementById('recommendation-team-size');
    const recommendationTableBody = document.getElementById('recommendation-table-body');

    // Update team size display
    teamSize.addEventListener('input', function() {
        teamSizeValue.textContent = this.value;
    });

    // Add a new skill row
    addSkillBtn.addEventListener('click', function() {
        const newSkillRow = document.createElement('div');
        newSkillRow.className = 'skill-row mb-2';
        newSkillRow.innerHTML = `
            <div class="row align-items-center">
                <div class="col-6">
                    <input type="text" class="form-control skill-name" placeholder="Skill name" required>
                </div>
                <div class="col-5">
                    <label>Importance: <span class="importance-value">0.7</span></label>
                    <input type="range" class="form-range skill-importance" min="0" max="1" step="0.1" value="0.7">
                </div>
                <div class="col-1">
                    <button type="button" class="btn btn-danger btn-sm remove-skill">✕</button>
                </div>
            </div>
        `;
        skillsContainer.appendChild(newSkillRow);

        // Add event listener to update importance value
        const importanceSlider = newSkillRow.querySelector('.skill-importance');
        const importanceValue = newSkillRow.querySelector('.importance-value');
        importanceSlider.addEventListener('input', function() {
            importanceValue.textContent = this.value;
        });

        // Add event listener to remove skill button
        const removeBtn = newSkillRow.querySelector('.remove-skill');
        removeBtn.addEventListener('click', function() {
            skillsContainer.removeChild(newSkillRow);
        });
    });

    // Add event listeners to existing skill importance sliders
    document.querySelectorAll('.skill-importance').forEach(slider => {
        slider.addEventListener('input', function() {
            this.parentElement.querySelector('.importance-value').textContent = this.value;
        });
    });

    // Add event listeners to existing remove skill buttons
    document.querySelectorAll('.remove-skill').forEach(button => {
        button.addEventListener('click', function() {
            const skillRow = this.closest('.skill-row');
            skillsContainer.removeChild(skillRow);
        });
    });

    // Add a new role row
    addRoleBtn.addEventListener('click', function() {
        const newRoleRow = document.createElement('div');
        newRoleRow.className = 'role-row mb-2';
        newRoleRow.innerHTML = `
            <div class="row align-items-center">
                <div class="col-8">
                    <select class="form-select role-name">
                        <option value="Developer">Developer</option>
                        <option value="Designer">Designer</option>
                        <option value="Project Manager">Project Manager</option>
                        <option value="QA Engineer">QA Engineer</option>
                        <option value="DevOps Engineer">DevOps Engineer</option>
                    </select>
                </div>
                <div class="col-3">
                    <input type="number" class="form-control role-count" min="1" max="10" value="1">
                </div>
                <div class="col-1">
                    <button type="button" class="btn btn-danger btn-sm remove-role">✕</button>
                </div>
            </div>
        `;
        rolesContainer.appendChild(newRoleRow);

        // Add event listener to remove role button
        const removeBtn = newRoleRow.querySelector('.remove-role');
        removeBtn.addEventListener('click', function() {
            rolesContainer.removeChild(newRoleRow);
        });
    });

    // Add event listeners to existing remove role buttons
    document.querySelectorAll('.remove-role').forEach(button => {
        button.addEventListener('click', function() {
            const roleRow = this.closest('.role-row');
            rolesContainer.removeChild(roleRow);
        });
    });

    // Form submission
    projectForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Disable submit button and show loading
        submitButton.disabled = true;
        submitButton.innerHTML = '<span id="loading-spinner"></span> Loading...';
        
        // Get form data
        const formData = {
            project_name: projectName.value,
            team_size: parseInt(teamSize.value),
            required_skills: [],
            required_roles: [],
            existing_team: []
        };
        
        // Get skills
        document.querySelectorAll('.skill-row').forEach(row => {
            const skillName = row.querySelector('.skill-name').value;
            const importance = parseFloat(row.querySelector('.skill-importance').value);
            
            if (skillName) {
                formData.required_skills.push({
                    skill_name: skillName,
                    importance: importance
                });
            }
        });
        
        // Get roles
        document.querySelectorAll('.role-row').forEach(row => {
            const roleName = row.querySelector('.role-name').value;
            const count = parseInt(row.querySelector('.role-count').value);
            
            if (roleName) {
                formData.required_roles.push({
                    role_name: roleName,
                    count: count
                });
            }
        });
        
        try {
            // Send API request
            const response = await fetch('http://localhost:8000/recommend', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });
            
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Display recommendation
            displayRecommendation(data);
            
        } catch (error) {
            console.error('Error:', error);
            alert('Error getting recommendations. Please check if the API is running.');
        } finally {
            // Re-enable submit button
            submitButton.disabled = false;
            submitButton.textContent = 'Get Recommendations';
        }
    });

    // Display recommendation
    function displayRecommendation(data) {
        // Update project information
        recommendationProjectName.textContent = data.project_name;
        recommendationTeamSize.textContent = data.team_size;
        
        // Clear existing recommendations
        recommendationTableBody.innerHTML = '';
        
        // Add team members to table
        data.team.forEach(member => {
            const row = document.createElement('tr');
            
            // Create score percentage
            const scorePercent = Math.round(member.score * 100);
            
            row.innerHTML = `
                <td>${member.resource_id}</td>
                <td>
                    <div class="d-flex align-items-center">
                        <div class="score-bar me-2" style="width: ${scorePercent}px"></div>
                        <span>${scorePercent}%</span>
                    </div>
                </td>
                <td>${member.explanation}</td>
            `;
            
            recommendationTableBody.appendChild(row);
        });
        
        // Show recommendation container
        recommendationContainer.classList.remove('d-none');
        
        // Scroll to recommendation
        recommendationContainer.scrollIntoView({ behavior: 'smooth' });
    }
});
