# PIRvision Occupancy Detection Using LSTM
This project demonstrates the development of an LSTM-based model for occupancy detection using the PIRvision dataset. The system analyzes sensor data to determine room occupancy, utilizing deep learning techniques with comprehensive validation and evaluation procedures.

## 📋 Overview

The project implements a complete machine learning pipeline for occupancy detection:
- Data preprocessing and exploratory analysis
- LSTM model architecture implementation
- 5-fold cross-validation training methodology
- Checkpoint creation for model persistence
- Comprehensive evaluation metrics
- Performance visualization

## 🧠 Model Architecture

The system uses a Long Short-Term Memory (LSTM) neural network specifically designed for sequential sensor data analysis. The model processes normalized PIR sensor readings to classify occupancy states with high accuracy.

## 📊 Performance Metrics

The evaluation provides detailed performance analytics:
- Mean accuracy across all folds
- Standard deviation of accuracy
- Macro F1-score
- Per-class precision, recall, and F1 scores
- Confusion matrix visualization
- Training/validation loss and accuracy curves

## 🔧 Dependencies

The project requires the following libraries:

```
Python 3.x
Pandas
NumPy
PyTorch
scikit-learn
Matplotlib
```

## ⚙️ Installation

You can install the necessary libraries using pip:

```bash
pip install pandas numpy torch scikit-learn matplotlib
```

Alternatively, if you use Anaconda:

```bash
conda install pandas numpy scikit-learn matplotlib
conda install pytorch torchvision torchaudio -c pytorch
```

## 📁 Project Structure

The project is organized in a Jupyter Notebook with the following sections:

1. **Imports & Global Settings**: Libraries and device configuration
2. **Data Loading & Preprocessing**: Dataset preparation and normalization
3. **Model Definition**: PIRvisionLSTM architecture and initialization
4. **5-Fold Cross-Validation Training**: Training implementation with checkpoint creation
5. **Evaluation Function**: Implementation for model assessment on test data

## 🚀 Running the Code

### Execute Cells in Order

1. Run the cell with library imports and global settings
2. Continue through each cell in sequence:
   - Data Loading & Preprocessing
   - Helper Functions and Model Definition
   - 5-Fold Cross-Validation Training (this will also save checkpoints)
   - Evaluation Function and its usage
3. Ensure all cells are executed to avoid missing dependencies or definitions

### Evaluating the Model

After training is completed and the checkpoint is saved (e.g., 'fold3_checkpoint.pth'), run the evaluation cell that calls the `evaluate_model` function. The function will output the accuracy of the saved model on the specified dataset.

## 📝 Additional Notes

- Ensure that the dataset file 'pirvision_office_dataset1.csv' is located in the same directory as your notebook or provide the correct path
- If you modify any parameters (hyperparameters or file paths), update them consistently throughout the notebook
- The evaluation function currently re-fits the scalers on the evaluation data. For a more consistent evaluation, you may want to save and reuse the scalers from the training phase


## 🔄 Future Improvements

- Implementation of additional neural network architectures
- Hyperparameter optimization
- Expanded feature engineering
- Real-time inference capabilities

## 📄 License

[MIT License](LICENSE)
