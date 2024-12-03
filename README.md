# Timeseries Energy Consumption

Using a temporal convolutional network to classify future energy consumption

Original dataset can be found here: https://huggingface.co/datasets/EDS-lab/electricity-demand

## Installation Setup

- Hardware
  - Developed under Windows with Nvidia 3080ti
- Software
  - Python 3.9.2
    - If using `pyenv`, use:
      ```bash
      pyenv install 3.9.2
      pyenv local 3.9.2
      ```
    - **Install dependencies**:
      ```bash
      pip install --upgrade pip
      pip install -r requirements.txt
      ```
    - **Or install dependencies manually, as long as Python==3.9.2**:
      ```bash
      pip install --upgrade pip
      pip install keras_tcn==3.5.0 matplotlib==3.9.3 numpy==2.1.3 pandas==2.2.3 scikit_learn==1.5.2 tensorflow==2.10.1 tensorflow-addons==0.22.0 pyarrow==18.0.0 jupyter
      ```

## Usage and Results

### There are three main notebooks: `TCN.ipynb`, `expirement.ipynb`, `data_handeling.ipynb`

- To quickly see the results and see accuracy, precision, recall and F-1 score head to `expirement.ipynb`.
- To see the source as to how the network was trained or feature engineering head to `TCN.ipynb`.
- To see how the data was handeled and resampled head to `data_handeling.ipynb`.

Please be aware the original parquet files are not located in this repository thus they need to be downloaded before running data_handeling.

### To reproduce the results
- Open `experiment.ipynb` and run the cells to reproduce the experiment results.
  - It will be setup to run the final expirement results which will be the `tcn_model_0.8_4.h5`.
  - Produces the following metrics accuracy, precision, recall, and F-1 for each class label. Additionally display a confusion matrix.
  
## **File & Directory Structure**
- **`TCN.ipynb`: source code for how the model was trained and feature engineering**
- **`experiment.ipynb`: the main experiment script to see results of models**
- **`data_handeling.ipynb`: data script that was used for resampling. To run this script please download the original set from huggingface linnked at the top**
- `requirements.txt`: contains all the necessary libraries that need to be install to run the project
- `data/`: core datasets not here but can be found in the huggingface link
  - `demand_weather_merged_0.5.parquet`: Combined datasets. Sampled at 50% of total data due to memory constraints.
  - `demand_weather_merged_0.5.parquet`: Combined datasets. Sampled at 80% of total data.
- `logging_images/`: containing all training images
  - `other_test_imgs/`: images from previous training that was worth keeping
    - `0.5_4/`: 50% dataset at 4 weeks
      - `acc_vs_epoch_0.5_4.png`
      - `loss_vs_epoch_0.5_4.png`
    - `0.5_12/`: 50% dataset at 12 weeks
      - `acc_vs_epoch_0.5_12.png`
      - `loss_vs_epoch_0.5_12.png`
    - `0.8_12/`: 80% dataset at 12 weeks
      - `acc_vs_epoch_0.8_12.png`
      - `loss_vs_epoch_0.8_12.png`
  - `classification_metrics_and_report.png`: final expirement metrics
  - `confusion_matrix.png`: final expirement confusion matrix
  - `acc_vs_epoch_0.8_4.png`: final expirement accuracy vs epoch test
  - `loss_vs_epoch_0.8_4.png`: final expirement loss vs epoch test
- `saved_model/`: contains the final model and old saved models
  - `old_models/`: previous models that were worth saving
    - `tcn_model_0.5_4.h5`
    - `tcn_model_0.5_12.h5`
    - `tcn_model_0.8_12.h5`
  - `tcn_model_0.8_4.h5`: model used for expirement and project results
- `README.md`
- `.gitignore`

