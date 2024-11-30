# Scam Call Detection Through Deep Learning Milestone 1

This Milestone has taken a shift from using the pretrained Resnet50 models to a custom CNN. The change allows further tuning and specificity within the model. This milestone also includes an updated dataset (120 examples -> 1110 examples), and a rudimentry version of the application used to showcase the models that will be built later. This rudimentry version includes a pickled version of the model and a system to intake an audio file, transcribe it, create the FFT spectogram, create the MFCC spectogram and finally make a prediction.

The results of the model training are displayed in the PDF below and usage of the pickled model is shown in the video.

## Video

https://www.youtube.com/watch?v=B812UWL9WPs&ab_channel=SheyBarpagga

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

    - Install PyTorch with CPU support, torchvision, and torchtext (Please use the recommended versions):

      ```bash
      pip install torch==2.2.0+cpu torchvision==0.17.0+cpu torchtext==0.16.2 -f https://download.pytorch.org/whl/torch_stable.html
      ```

    - Install librosa:

      ```bash
      pip install librosa
      ```

    - Install matplotlib:

      ```bash
      pip install matplotlib
      ```

    - Install transformers:
      ```bash
      pip install transformers
      ```

    - Install NumPy:
      ```bash
      pip install numpy<2
      ```

    - Alternatively, use the provided requirements.txt file to install the needed packages:
      ```bash
      pip install -r requirements.txt
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
```