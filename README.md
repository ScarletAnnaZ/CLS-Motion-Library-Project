# Motion Library Project 


## Overview 
This project is part of a research collaboration focused on real-time human-robot interaction in improvisational dance. The main purpose of this study is to develop a virtual system that enables virtual agents to recognize human dance movements using motion capture data and generate meaningful dance responses in time. The system is based on the motion library architecture, which uses machine learning models to classify input actions and trigger corresponding agent responses according to a preset response strategy. The final result can be visualized in 3D in Blender.

## Installation

1. This project requires Python 3.11+ and Blender 4.3+. You can install Python from python.org（https://www.python.org/downloads/） and Blender from blender.org(https://www.blender.org/download/).

2. Install required Python packages: Make sure your pip is up to date:
   ```bash
   pip install --upgrade pip
   ```

3. Then install dependencies from the requirements.txt file(The 'requirements.txt' file is located in the root directory of this repository. It contains all necessary Python packages to run the project.  )
   ```bash
   pip install -r requirements.txt
   ```
## Start The Project
 ### 1. Prepare dataset

This project operates on motion capture data in '.bvh' format. To use your own data or the provided samples, follow the steps below.

1. Place raw '.bvh' files: Create a 'data/' directory in the root of the project (if it doesn't already exist), and place your raw motion files there. (The data project used is from CMU Dataset. If you want to use it as well, you can find:https://mocap.cs.cmu.edu/)
2. Use scripts in scripts/ to:

    - Standardise the label file
    - Standardise the motion data to the specific frame size segments
    - Extract features from segments
  
### 2. Train the models
Train classification models (KNN / RandomForest): Model outputs will be stored in the output/models/ directory. 

### 3. Run prediction
Run inference on new .bvh input files using a trained model, and get the predicted labellist.

### 4. Generate response
Set the response mapping strategy, and map predicted motion labels to agent response motions based on the strategy. Select the motion segments from the library and then merge them as the output response motion.

### 5. Visualization
Open Blender and import the output motion to visualize the results.

### Extra experiments
- The project also utilises Bayesian Optimisation for hyperparameter tuning.
- Sensitivity analysis -- explore the influence on both KNN and RF models with different segment sizes (frame sizes[10,30,60...])

## Quick review
You can try the script of 'scrpit2/run_pipeline.py'. This script combines the process together with prediction labels, using response strategies, selects and merges motion segments. You can quickly change the path of input data and the model you want to use to experiment and understand the progress of the project.

### ⚠️ Note
Due to GitHub's file size limitations, this repository does not include the original motion capture datasets (BVH files), the processed motion library, extracted feature files, or trained model files. Before running the project, please manually import the required datasets and complete the following steps:

- Data preprocessing and motion library construction

- Feature extraction and model training

Only after completing these steps should you run the main pipeline script 'script2/run_pipeline.py' for inference or demonstration.
