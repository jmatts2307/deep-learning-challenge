# Alphabet Soup Neural Network Analysis

## Overview of the Analysis
The purpose of this analysis is to build a binary classification model using deep learning to predict whether an applicant would be successful in securing funding from Alphabet Soup. The dataset contains information on over 34,000 organizations, with features describing application types, affiliations, classifications and financial details. 

To achieve this goal, we:

- Preprocessed the data by handling categorical features, scaling numerical values, and dropping unnecessary columns.

- Designed a deep neural network model using TensorFlow/Keras.

- Attempted different optimizations to improve accuracy beyond 75%.

- Evaluated the model's performance and considered alternative approaches.


## Results
Data Processing

Target Variables (y):

- IS_SUCCESSFUL - Indicates if the applicant successfully used the funding.

Feature Variables (X):

- All remaining columns except identification variables.

- Categorical variables were one-hot encoded (APPLICATION_TYPE, CLASSIFICATION, AFFILIATION).

- Numerical variables were scaled using StandardScaler().

    
Variables Removed:

- EIN and NAME were removed since they contained no predictive value.

- Some categorical values with low frequency were grouped into an 'Other' category for better organization.

Compiling, Training, and Evaluating the Model:

Inital Model:

- 2 hidden layers (80 neurons and 30 neurons)

- Activation Function: ReLu

- Output Layer: Sigmoid for binary classification


Final Optimized Model:

- 3 hidden layers (128 neurons, 64 neurons, 32 neurons)

- Activation Function: ReLu

- Dropout (0.3) to prevent overfitting (Added during optimization)

- Batch Normalization for smoother training

- Learning Rate: 0.0005 for finer updates

- Epochs increased to 300 for longer training

Model Performance:

- Initial Accuracy: 72.7%

- Final Optimized Accuracy: 72.9%

- Target Accuracy 75%: Not Reached

Optimization Attempts:

Optimization 1: Increased neurons in hidden layers and added additional layer. This attempt was to extract better patterns but ended in 72.5% accuracy. 

Optimization 2: Changed activation from ReLu to tanh, kept additional layers. This result ended in 72.5%.

Optimization 3: Kept ReLu activation, applied 0.3 dropout, added batch normalization to stabilize learning, and increased epochs to 300 to allow deeper learning. This result ended in almost no impact with accuracy of 72.9%.  

## Summary

Final Accuracy: 72.9%

Challenges:

- Model performance plateaued even with different optimizations.

- Some features may have added noise instead of value.

- A different approach may be needed to reach 75% or higher accuracy.

Recommendations:

Random Forest or XGBoost

- Tree-based models handle categorical data without needing one-hot encoding.

- Feature importance ranking can help determine which features matter most.

- Better for structure or tabular data like our dataset.

Although deep learning projects are powerful, they are not always the best choice for structured/tabular data. Tree-based models like XGBoost may outperform a neural network in this case. 
