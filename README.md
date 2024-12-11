# Scam Call Detection Through Deep Learning Milestone 2

This Milestone has taken a shift from using the pretrained Resnet50 models to a custom CNN. The change allows further tuning and specificity within the model. This milestone also includes an updated dataset (120 examples -> 1110 examples), and a rudimentry version of the application used to showcase the models. 

The results of the model training are displayed in the PDF below and usage of the pickled model is shown in the video.

## Video

https://youtu.be/PaVh1AgpzRE

## Report

The report can be accessed here:

https://www.overleaf.com/read/djnwgjtwmntf#772811

## Environment Setup

To run the project, you'll need to set up a conda environment and install the required dependencies. Follow the steps below:

### Step 1: Create and Set Up the Conda Environment

1. **Create a new conda environment**  
   Use the following command to create a new conda environment. Replace `env_name` with your preferred environment name:

    ```bash
    conda create --name env_name
    ```

2. **Activate the environment**  
   Once the environment is created, ensure you activate it before installing the dependencies:

    ```bash
    conda activate env_name
    ```

3. **Install the required packages**  
   After activating the environment, install the necessary packages with the following commands:

    - Use the provided requirements.txt file to install the needed packages:
      ```bash
      pip install -r requirements.txt
      ```


### Step 2: Navigate to the project

2. **Navigate to the project folder**  
   Navigate to the folder where the repository was downloaded, specifically into the server folder:

    ```bash
    cd path/to/project-folder/server
    ```

### Step 3: Run the Project

Once inside the project folder, ensure your environment is activated, and run the `main.py` file using the following command:

```bash
python main.py
```

You can access the webpage by visiting http://127.0.0.1:8000/
