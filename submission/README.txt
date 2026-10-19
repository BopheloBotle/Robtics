============================================================
  SUBMISSION NOTES — ASR 700 Robotics Project
============================================================

This submission contains the complete Intelligent Autonomous
Robotic Systems Project for ASR 700.

CONTENTS
--------
- component_a/    : Component A source, worlds, controller, tests, results
- component_b/    : Component B source, worlds, controller, tests, results
- report/         : Markdown report source and PDF (if generated)
- README.txt      : Full setup and usage instructions
- requirements.txt: Python dependencies
- run_all.py      : Run all offline experiments
- validate_project.py: Project validation script

QUICK START
-----------
1. Install dependencies:   pip install -r requirements.txt
2. Run offline pipelines:  python run_all.py
3. Run tests:              python -m pytest component_a/tests/ component_b/tests/ -v
4. Validate:               python validate_project.py
5. Open Webots Component A: component_a/worlds/component_a_world.wbt
6. Open Webots Component B: component_b/worlds/component_b_world.wbt

See README.txt for full instructions.

TECHNOLOGY
----------
Simulator: Webots R2025a (NOT ROS, NOT Gazebo)
Language:  Python 3.13.5
============================================================
