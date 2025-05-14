## Setup Instructions

1. **Create a Conda environment**  
   Run the following command in your terminal to create a new environment named `RoboA` with Python 3.12.8:

   ```bash
   conda create -n RoboA python=3.12.8

2. **Activate the environment and install dependencies**

    Once the environment is created, activate it and install the required libraries:

    ```bash
    conda activate RoboA
    pip install -r requirements.txt

3. **Select the Conda environment as the Jupyter kernel**

    Open the Jupyter Notebook, and change the kernel to use the RoboA environment.

4. **(Optional) Fix yfinance errors**

    If you encounter issues with yfinance, update it using:
    
    ```bash
    pip install yfinance --upgrade

