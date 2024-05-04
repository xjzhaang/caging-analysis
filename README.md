# Nuclei caging analysis

This repository contains an interactive Streamlit application that analyzes nuclei caging on microgrooves.
You can run the repository locally, or find it online at https://caging.streamlit.app/.

Local use is highly recommended.

## Installation

Clone this repository to your local machine, or simple download and extract the folder:

```bash
git clone https://github.com/xjzhaang/caging-analysis.git

#Download link
https://github.com/xjzhaang/caging-analysis/archive/refs/heads/main.zip
```

### Navigate to the cloned directory
```bash
cd caging-analysis
```

### Install Environment 
#### Using Conda

If you prefer using Conda for managing environments, you can create a Conda environment and install the dependencies.
Create a Conda environment:

```bash
conda create --name caging python=3.11
conda activate caging
pip install -r requirements.txt
```

#### Without Conda

If you prefer not to use Conda, you can create a virtual environment using Python's built-in venv module.
Create a virtual environment:
```bash
python -m venv caging

#Activate the virtual environment (Windows)
source caging/bin/activate
#Activate the virtual environment (Unix or MacOS)
source caging/bin/activate

pip install -r requirements.txt
```


## Usage
To run the Streamlit app, execute the following command:
```bash
streamlit run app.py
```