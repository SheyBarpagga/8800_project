# Scam Call Detection Through Deep Learning Prototype

This prototype utilizes Whisper and Resnet50. The approach is a multi-input model that takes in 2 images (transformed into RGB values as per Resnet50's requirements), the transcription of the call (provided by Whisper), and a binary label indicating if it is a scam or not. The prediction uses BCELoss, a criterion that measues cross entropy between the target and input probabilities to determine an output. 

The model achieved 86.36% accuracy on 10 epochs. 

## PDF Document + Video
Because currently it takes about 20 minutes to run the program I have provided a screen shot of the run in the PDF file linked below:

https://drive.google.com/file/d/10N3UBzspwU2LJsbwHgEQU-CD9F-ksv4V/view?usp=sharing

## Environment Setup

To run the project, you'll need to set up a conda environment and install the required dependencies. Follow the steps below:

### Step 1: Create and Set Up the Conda Environment

1. **Create a new conda environment**  
   Use the following command to create a new conda environment. Replace `env_name` with your preferred environment name:

    ```bash
    conda create --name env_name python=3.9
    ```

2. **Activate the environment**  
   Once the environment is created, ensure you activate it before installing the dependencies:

    ```bash
    conda activate env_name
    ```

3. **Install the required packages**  
   After activating the environment, install the necessary packages with the following commands:

    - Install PyTorch with CPU support, torchvision, and torchaudio:

      ```bash
      conda install pytorch torchvision torchaudio cpuonly -c pytorch
      ```

    - Install torchtext for text processing:

      ```bash
      conda install pytorch::torchtext
      ```

    - Install pandas for data manipulation:

      ```bash
      conda install pandas
      ```

    - Install scikit-learn for utilities:

      ```bash
      conda install conda-forge::scikit-learn
      ```

    - Install napari-skimage-regionprops to get access to skimage:

      ```bash
      conda install conda-forge::napari-skimage-regionprops
      ```

### Step 2: Pull the GitHub Repository

1. **Clone the repository**  
   Pull the project from GitHub:

    ```bash
    git clone https://github.com/SheyBarpagga/8800_project.git
    ```

2. **Navigate to the project folder**  
   Navigate to the folder where the repository was downloaded:

    ```bash
    cd path/to/project-folder
    ```

### Step 3: Run the Project

Once inside the project folder, ensure your environment is activated, and run the `main.py` file using the following command:

```bash
python main.py