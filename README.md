# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS
### STEP 1: 

Create your dataset in a Google sheet with one numeric input and one numeric output.

### STEP 2: 

Split the dataset into training and testing

### STEP 3: 

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4: 

Build the Neural Network Model and compile the model.

### STEP 5: 

Train the model with the training data.

### STEP 6: 

Plot the performance plot

### STEP 7: 

Evaluate the model with the testing data.

### STEP 8: 

Use the trained model to predict  for a new input value .

## PROGRAM

### Name:

### Register Number:

```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here

```

### Dataset Information
Include screenshot of the generated data

### OUTPUT
### LOSS:
<img width="492" height="247" alt="image" src="https://github.com/user-attachments/assets/6975b781-fc5a-4b28-8766-3401f3f54bf8" />
<img width="347" height="59" alt="image" src="https://github.com/user-attachments/assets/642c9c0d-92f4-469a-9d8a-b303172b84b8" />

### Training Loss Vs Iteration Plot
<img width="732" height="585" alt="image" src="https://github.com/user-attachments/assets/8859733a-f1b7-41d3-a89d-cd3b2b6e14b4" />

### New Sample Data Prediction
<img width="344" height="42" alt="image" src="https://github.com/user-attachments/assets/e9c8c160-cc0f-479b-a995-09671c913d9c" />

## RESULT
Thus, a neural network regression model was successfully developed and trained using PyTorch.
