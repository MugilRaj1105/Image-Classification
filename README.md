# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset

The goal of this project is to develop a **Convolutional Neural Network (CNN)** for image classification using the **Fashion-MNIST** dataset. The Fashion-MNIST dataset contains images of various clothing items (T-shirts, trousers, dresses, shoes, etc.), and the model aims to classify them correctly. The challenge is to achieve **high accuracy** while maintaining **efficiency**.
## Neural Network Model

![425840736-6acab57a-cf5e-4963-a584-024b1d03e3e9](https://github.com/user-attachments/assets/2a4ae2ee-eccd-4f09-8dcb-19aa85efc78a)

## DESIGN STEPS

#### STEP 1: Problem Statement  
Define the objective of classifying fashion items (T-shirts, trousers, dresses, shoes, etc.) using a **Convolutional Neural Network (CNN)**.  

#### STEP 2: Dataset Collection  
Use the **Fashion-MNIST dataset**, which contains **60,000** training images and **10,000** test images of various clothing items.  

#### STEP 3: Data Preprocessing  
Convert images to tensors, normalize pixel values, and create **DataLoaders** for batch processing.  

#### STEP 4: Model Architecture  
Design a CNN with **convolutional layers**, **activation functions**, **pooling layers**, and **fully connected layers** to extract features and classify clothing items.  

#### STEP 5: Model Training  
Train the model using a suitable **loss function** (**CrossEntropyLoss**) and **optimizer** (**Adam**) for multiple epochs.  

#### STEP 6: Model Evaluation  
Test the model on unseen data, compute **accuracy**, and analyze results using a **confusion matrix** and **classification report**.  

#### STEP 7: Model Deployment & Visualization  
Save the trained model, visualize predictions, and integrate it into an application if needed.  


## PROGRAM

### Name:MUGIL RAJ S A
### Register Number:212223220062
```python

class CNNClassifier(nn.Module):
  def __init__(self): # Define __init__ method explicitly
    super(CNNClassifier, self).__init__() # Call super().__init__() within __init__
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1) # Correct argument names
    self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)  # Correct argument names
    self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1) # Correct argument names
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(128 * 3 * 3, 128) # Adjust input size for Linear layer (Calculation needs update if image size changed)
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 10)

  def forward(self, x):
    x = self.pool(torch.relu(self.conv1(x))) # Correctly call self.conv1
    x = self.pool(torch.relu(self.conv2(x)))  # Correctly call self.conv2
    x = self.pool(torch.relu(self.conv3(x))) # Correctly call self.conv3
    x = x.view(x.size(0), -1) # Flatten the tensor
    x = torch.relu(self.fc1(x)) # Correctly call self.fc1
    x = torch.relu(self.fc2(x)) # Correctly call self.fc2
    x = self.fc3(x)
    return x


```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(), lr=0.001)
```

```python
## Step 3: Train the Model
def train_model(model, train_loader, optimizer, criterion, num_epochs=3):
    print('Name: MUGIL RAJ S A')
    print('Register Number: 212223220062')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Print only once per epoch
        avg_loss = running_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
```

## OUTPUT
### Training Loss per Epoch
<img width="302" height="121" alt="image" src="https://github.com/user-attachments/assets/de4d4e30-4ff4-4a11-aad5-3b46d60c77c4" />


### Confusion Matrix
<img width="1057" height="803" alt="image" src="https://github.com/user-attachments/assets/0a190daf-c222-4a09-a5f8-fc3770042660" />




### Classification Report
<img width="587" height="422" alt="image" src="https://github.com/user-attachments/assets/054240f2-ee5d-4988-ab35-2692a31896f7" />



### New Sample Data Prediction

<img width="607" height="642" alt="image" src="https://github.com/user-attachments/assets/86bf6019-331c-4af7-a0d5-66660c4a34c5" />


## RESULT
Thus ,the experiment was executed successfully.
