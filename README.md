# Developing a Neural Network Regression Model

## AIM
To develop a neural network regression model for the given dataset.

## THEORY
Explain the problem statement

## Neural Network Model
<img width="987" height="552" alt="image" src="https://github.com/user-attachments/assets/3d3b78de-7023-4ab0-8b7e-bb0018c446e7" />

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

### Name: Mohammad Suhael

### Register Number: 212224230164

```python
class NeuralNet(nn.Module):
  def __init__(self):
        super().__init__()
        # Include your code here
        self.fc1 = nn.Linear(1,8)
        self.fc2 = nn.Linear(8,10)
        self.fc3 = nn.Linear(10,1)
        self.relu = nn.ReLU()
        self.history = {'loss':[]}
  def forward(self,x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x



# Initialize the Model, Loss Function, and Optimizer
ai_brain = NeuralNet()
criterion = nn.MSELoss()
optimizer = optim.RMSprop(ai_brain.parameters(),lr = 0.001)


def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    # Write your code here
    for epoch in range(epochs):
      optimizer.zero_grad()
      loss = criterion(ai_brain(X_train),y_train)
      loss.backward()
      optimizer.step()

      ai_brain.history['loss'].append(loss.item())
      if epoch % 200 == 0:
          print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')
    

```

### Dataset Information
<img width="197" height="400" alt="image" src="https://github.com/user-attachments/assets/24a7b36f-2784-4660-ae1a-0f7727150ce0" />

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
