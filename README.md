# Robotic Arm RL Agent

## Files Included
- roboticarmagent.py - main script for training and evaluating the reinforcement learning robotic arm agent
- RoboticArmAgent.docx - report describing design, algorithms, implementation, and results

## How to Run

1) Install Python 3.11 and verify
python --version

2) Install required dependencies
pip install torch numpy gymnasium

3) Navigate to the project folder
cd path/to/your/project-folder

---

### Run WITHOUT visuals (faster, data-only training)

4) Train the agent
python roboticarmagent.py --mode train

5) Resume training if interrupted
python roboticarmagent.py --mode train --resume

6) Evaluate the trained agent (numerical results only)
python roboticarmagent.py --mode eval

---

### Run WITH visuals (MuJoCo simulation)

7) Train with visual environment
python roboticarmagent.py --mode train --mujoco

8) Evaluate with rendering to watch the agent
python roboticarmagent.py --mode eval --mujoco --render
