============================================================
  INTELLIGENT AUTONOMOUS ROBOTIC SYSTEMS PROJECT
  ASR 700 — University Assignment Submission
============================================================

PROJECT TITLE  : Intelligent Autonomous Robotic Systems Project
MODULE         : Autonomous Systems and Robotics 700 (ASR 700)
SUBMISSION DATE: October 2026
NOTE           : ROS and Gazebo are NOT used anywhere in this project.
                 All simulation uses Webots R2025a with Python 3.13.

------------------------------------------------------------
  PROJECT OVERVIEW
------------------------------------------------------------
This project implements two integrated intelligent robotic systems:

  Component A — Autonomous Mobile Robot
    A single differential-drive robot that maps its environment using
    LiDAR and a camera, plans a path using Dijkstra's algorithm and A*,
    and follows the path using a PID controller.

  Component B — Cooperative Swarm Robotics
    A swarm of three autonomous robots that communicate via Webots
    Emitter/Receiver devices, reach consensus on shared state, allocate
    tasks using Particle Swarm Optimisation (PSO), and navigate using a
    trained neural network (NumpyMLP).

------------------------------------------------------------
  KEY RESULTS (offline experiments)
------------------------------------------------------------
  Component A
    Grid         : 100x100 cells, 0.10m resolution, 10m x 10m arena
    Dijkstra     : 9555 nodes explored, 10.72m path, ~25ms planning
    A*           : 715 nodes explored, 10.72m path, ~2ms planning
    PID          : heading converges to <5 deg error (well-tuned gains)
    Navigation   : goal reached in offline simulation

  Component B
    NumpyMLP     : 95.50% test accuracy (400 test samples, balanced)
    scikit-learn : 95.25% test accuracy
    Consensus    : converged in 3 iterations (alpha=0.3, ring topology)
    PSO cost     : 12.58m  vs  Greedy cost: 13.65m (PSO 7.8% better)

------------------------------------------------------------
  FOLDER STRUCTURE
------------------------------------------------------------
Robotics/
  README.txt                  <- This file
  requirements.txt            <- Python package dependencies
  .gitignore
  run_all.py                  <- Run all offline experiments
  run_component_a.py          <- Run Component A offline pipeline
  run_component_b.py          <- Run Component B offline pipeline
  validate_project.py         <- Full project validation script

  component_a/
    worlds/
      component_a_world.wbt   <- Webots world (open in Webots to simulate)
    controllers/
      autonomous_controller/
        autonomous_controller.py  <- Webots robot controller
    src/
      configuration.py        <- All Component A parameters
      occupancy_grid.py       <- 2D occupancy grid with coordinate conv.
      dijkstra.py             <- Dijkstra's shortest-path algorithm
      astar.py                <- A* with octile heuristic
      robot_model.py          <- Differential-drive kinematics
      pid_controller.py       <- PID with anti-windup + derivative filter
      sensors.py              <- Synthetic LiDAR and camera generators
      perception.py           <- LiDAR and camera processing pipeline
      sensor_fusion.py        <- Evidence-accumulation sensor fusion
      navigation.py           <- Waypoint-following navigation controller
      evaluation.py           <- Offline experiments + figure generation
    data/raw/                 <- Raw sensor logs (created by Webots run)
    data/processed/           <- Processed datasets
    results/figures/          <- All 7 required Matplotlib figures (PNG)
    results/metrics/          <- CSV and JSON metric summaries
    results/logs/             <- Webots simulation logs
    tests/
      test_pathfinding.py     <- 37 tests for Dijkstra and A*
      test_pid.py             <- 13 tests for PID controller
      test_occupancy_grid.py  <- 15 tests for occupancy grid

  component_b/
    worlds/
      component_b_world.wbt   <- Webots swarm world (3 robots)
    controllers/
      swarm_controller/
        swarm_controller.py   <- Webots swarm controller (shared by all 3)
    src/
      configuration_b.py      <- All Component B parameters
      agents.py               <- Agent state and message data structures
      communication.py        <- Emitter/Receiver + LocalBus for offline
      consensus.py            <- Average consensus algorithm
      particle_swarm.py       <- PSO for task allocation
      task_allocation.py      <- PSO and greedy allocation + comparison
      neural_network.py       <- NumpyMLP (pure NumPy feed-forward NN)
      training.py             <- Dataset generation + training pipeline
      evaluation.py           <- Offline experiments + figure generation
    data/training/            <- Balanced synthetic training dataset (CSV)
    data/evaluation/          <- Evaluation dataset
    models/                   <- Saved NumpyMLP (.npz) and sklearn (.pkl)
    results/figures/          <- All 6 required Component B figures (PNG)
    results/metrics/          <- CSV and JSON metric summaries
    results/logs/             <- Webots swarm logs (created by Webots run)
    tests/
      test_consensus.py       <- 13 tests for consensus algorithm
      test_task_allocation.py <- 19 tests for PSO + greedy allocation
      test_neural_network.py  <- 23 tests for NumpyMLP + teacher policy

  report/
    report_source.md          <- Editable Markdown report source
    report.pdf                <- PDF report (generate from .md — see below)

  submission/
    README.txt                <- Submission instructions

------------------------------------------------------------
  SYSTEM REQUIREMENTS
------------------------------------------------------------
  Operating system : Windows 11
  Python           : 3.13.5
  Webots           : R2025a
                     Executable: C:\Program Files\Webots\msys64\mingw64\bin\webots.exe
  RAM              : >= 8 GB recommended
  Disk             : >= 500 MB free

------------------------------------------------------------
  WINDOWS 11 SETUP INSTRUCTIONS
------------------------------------------------------------
1. Verify Python 3.13 is installed:
     python --version

2. Install Python dependencies:
     pip install -r requirements.txt

3. Verify Webots R2025a is installed:
     "C:\Program Files\Webots\msys64\mingw64\bin\webots.exe" --version

4. No virtual environment is required — packages are installed globally.
   If you prefer a virtual environment:
     python -m venv .venv
     .venv\Scripts\Activate.ps1
     pip install -r requirements.txt

------------------------------------------------------------
  HOW TO RUN THE OFFLINE EXPERIMENTS (no Webots needed)
------------------------------------------------------------
Run all Component A offline experiments (pathfinding, PID tuning,
navigation simulation, figures, metrics):
  python run_component_a.py

Run all Component B offline experiments (NN training, consensus,
PSO allocation, figures, metrics):
  python run_component_b.py

Run both together:
  python run_all.py

Run the full validation script:
  python validate_project.py

Run all automated unit tests:
  python -m pytest component_a/tests/ component_b/tests/ -v

------------------------------------------------------------
  HOW TO OPEN THE WEBOTS WORLDS AND RUN THE CONTROLLERS
------------------------------------------------------------
IMPORTANT: Webots must be launched manually from your desktop or
           Start Menu. The controllers start automatically when the
           simulation runs.

  Component A (autonomous navigation):
    1. Open Webots R2025a.
    2. File -> Open World...
    3. Navigate to: component_a\worlds\component_a_world.wbt
    4. Click the play button (or press Ctrl+T).
    5. The robot will:
       a. Build an occupancy grid from the world configuration.
       b. Plan a path using A* (Dijkstra as fallback).
       c. Follow the path using PID control.
       d. Save a log to component_a\results\logs\webots_run_log.csv
       e. Save a summary to component_a\results\metrics\webots_summary.json

  Component B (swarm robotics):
    1. Open Webots R2025a.
    2. File -> Open World...
    3. Navigate to: component_b\worlds\component_b_world.wbt
    4. Click the play button.
    5. All three robots start simultaneously. Each robot:
       a. Reads its ID from controllerArgs.
       b. Loads the PSO target assignment.
       c. Loads the trained NumpyMLP model.
       d. Navigates to its assigned target using the NN action.
       e. Saves a log to component_b\results\logs\webots_swarm_<id>.csv

  Note: The controllers use relative paths so they will work from any
        location as long as the project folder structure is preserved.

------------------------------------------------------------
  HOW TO GENERATE MATPLOTLIB FIGURES
------------------------------------------------------------
Figures are generated automatically by the offline pipeline scripts:
  python run_component_a.py   -> 7 figures in component_a/results/figures/
  python run_component_b.py   -> 6 figures in component_b/results/figures/

To regenerate figures without rerunning experiments, you can call the
evaluation modules directly:
  python -c "
  import sys; sys.path.insert(0,'component_a/src')
  from configuration import Config; from occupancy_grid import OccupancyGrid
  from evaluation import run_pathfinding_comparison, run_pid_tuning_experiment
  from evaluation import generate_all_component_a_figures
  cfg=Config(); g=OccupancyGrid(cfg); g.build_from_config()
  pr=run_pathfinding_comparison(g,cfg); pid=run_pid_tuning_experiment(cfg)
  generate_all_component_a_figures(g,pr,pid,cfg)
  "

------------------------------------------------------------
  HOW TO TRAIN AND EVALUATE THE NEURAL NETWORK
------------------------------------------------------------
Training is performed automatically by run_component_b.py.
To run training separately:
  python -c "
  import sys; sys.path.insert(0,'component_b/src')
  from configuration_b import ConfigB; from training import generate_training_data, train_and_evaluate_model
  cfg=ConfigB()
  generate_training_data(cfg)
  results = train_and_evaluate_model(cfg)
  print('Test accuracy:', results['accuracy'])
  "

Saved model artifacts:
  component_b/models/numpy_mlp.npz       <- NumpyMLP weights
  component_b/models/sklearn_mlp.pkl     <- scikit-learn MLPClassifier
  component_b/models/feature_scaler.json <- StandardScaler parameters
  component_b/data/training/nn_training_data.csv       <- Training data
  component_b/data/training/nn_training_metadata.json  <- Dataset info
  component_b/results/metrics/nn_evaluation.json       <- Eval results

------------------------------------------------------------
  HOW TO GENERATE THE PDF REPORT
------------------------------------------------------------
The report source is at: report/report_source.md

Option 1 — Pandoc (recommended):
  Install Pandoc from https://pandoc.org/installing.html
  Then run:
    pandoc report\report_source.md -o report\report.pdf --pdf-engine=xelatex

Option 2 — VS Code Markdown PDF extension:
  Install the "Markdown PDF" extension in VS Code.
  Open report\report_source.md and use the command
  "Markdown PDF: Export (pdf)".

Option 3 — Online converter:
  Paste the Markdown into https://md2pdf.netlify.app/ and download the PDF.

------------------------------------------------------------
  EXPECTED RUNTIME (estimates — actual time varies by hardware)
------------------------------------------------------------
  python run_component_a.py  : ~10–30 seconds
  python run_component_b.py  : ~30–90 seconds
  python -m pytest ...       : ~8–15 seconds
  python validate_project.py : ~15–30 seconds
  Webots Component A         : ~1–3 minutes real time
  Webots Component B         : ~2–5 minutes real time

------------------------------------------------------------
  OUTPUT LOCATIONS
------------------------------------------------------------
  Component A figures   : component_a/results/figures/*.png
  Component A metrics   : component_a/results/metrics/*.csv, *.json
  Component A Webots log: component_a/results/logs/webots_run_log.csv
  Component B figures   : component_b/results/figures/*.png
  Component B metrics   : component_b/results/metrics/*.csv, *.json
  Component B models    : component_b/models/*.npz, *.pkl, *.json
  Component B train data: component_b/data/training/
  Report                : report/report_source.md, report/report.pdf

------------------------------------------------------------
  TROUBLESHOOTING
------------------------------------------------------------
Problem: "No module named controller" inside Webots controller
Fix    : This is expected when running the controller outside Webots.
         Open the world in Webots — the controller runs automatically.

Problem: "No module named matplotlib" or similar import error
Fix    : Run: pip install -r requirements.txt

Problem: Webots world does not load EXTERNPROTO
Fix    : Ensure internet access is available (Webots downloads PROTO files
         from raw.githubusercontent.com on first load).

Problem: NN model not found when Webots swarm controller starts
Fix    : Run "python run_component_b.py" first to train and save the model.

Problem: Figures not generated
Fix    : Run "python run_component_a.py" and "python run_component_b.py".

Problem: validate_project.py reports FAIL for report files
Fix    : Run the report generation (see above) and ensure
         report/report_source.md exists.

------------------------------------------------------------
  KNOWN LIMITATIONS
------------------------------------------------------------
1. The Webots simulation requires a GUI — it cannot run headlessly on
   Windows without an active desktop session.

2. The offline navigation simulation uses a simplified kinematic model
   without real sensor feedback.  Actual Webots results may differ.

3. The consensus algorithm converges very quickly (3 iterations) because
   the initial state difference and alpha value are well-matched.  In
   a real distributed system, network latency would increase convergence time.

4. The PSO uses a continuous encoding with penalty for duplicate assignments.
   For very small swarms (3 agents), the Hungarian algorithm would be more
   efficient.

5. The neural network was trained on synthetic data from a rule-based
   teacher.  Performance on diverse real environments may be lower.

6. Camera perception uses simple thresholding.  Lighting changes in a
   real environment would require more robust vision algorithms.

------------------------------------------------------------
  TECHNOLOGY CONFIRMATION
------------------------------------------------------------
  Simulator : Webots R2025a  (NOT ROS, NOT Gazebo)
  Language  : Python 3.13.5
  Libraries : NumPy, Matplotlib, OpenCV, scikit-learn, SciPy
  ROS       : NOT USED
  Gazebo    : NOT USED
============================================================
