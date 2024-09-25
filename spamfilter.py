import numpy as np
from mat4py import loadmat

f = loadmat('spamdata.mat') # Loads mat file into a dictionary

k = 1 # Smoothing factor

# Training values (the y arrarys are array of arrays even though each inner array is just one value)
X = f["X"]
y = f["y"]

# Testing values
X2 = f["X2"]
y2 = f["y2"]

# Variables

pred = [] # Predictions for y (is/isnt spam)

# These arrays store probabilites for every X given a y

pX1gy1 = [] # Probility of X = 1 given y = 1
pX1gy0 = [] # Probility of X = 1 given y = 0
pX0gy1 = [] # Probility of X = 0 given y = 1
pX0gy0 = [] # Probility of X = 0 given y = 0

numx1y1 = 0 # Number of observed x = 1 when y = 1
numx1y0 = 0 # Number of observed x = 1 when y = 0

numy1 = 0 # Number of observed y = 1
numy0 = 0 # Number of observed y = 0

# Scan through data for above variables

for e in range(0,len(X[0])): # This will will run for every "word" X (word 1 of each email, word 2, etc...)

    for i, j in enumerate(X): # X is an array of array so this goes to the corrosponding word of each "email"
        
        if y[i][0] == 1: # Checks if the corrosponding y we are on is 1
            numy1 += 1
            if j[e] == 1: # Checks if current word we are evalating is 1
                numx1y1 += 1

        elif y[i][0] == 0: # Checks if the corrosponding y we are on is 0
            numy0 += 1
            if j[e] == 1: # Checks if current word we are evalating is 1
                numx1y0 += 1
    
    # This should go through each X by position and use the probability formula to generate
    # a probability given y for each "word" X.
                
    pX1gy1.append((numx1y1 + k)/(numy1 + 2*k))
    pX1gy0.append((numx1y0 + k)/(numy0 + 2*k))

    # By definition of a complemenet I can calculate the complements as 1 - previous probabilities

    pX0gy1.append(1-(numx1y1 + k)/(numy1 + 2*k))
    pX0gy0.append(1-(numx1y0 + k)/(numy0 + 2*k))

# Testing on test data

# Will go through every testing email and use previous calculated probabilities 
for i in range(0,len(X2)):
    py1gX = numy1/len(y2) # Start with probability of Y = 1

    for j in range(0,len(X2[i])): # Go through each X and multiply the relevant conditional probability
        if X2[i][j] == 1:
            py1gX = py1gX*pX1gy1[j]
        else:
            py1gX = py1gX*pX0gy1[j]

    # Appends a 1 or 0 prediction for each y
    if py1gX >= (1 - py1gX): # Probability of y = 0 given X's is the complement of y = 1 given X's
        pred.append(1)
    else:
        pred.append(0)

# Calculating Accurracy

correct = 0
for i, j in enumerate(pred): # Goes through an compares my predicitions to the actual y2 data
    if j == y2[i][0]:
        correct += 1

print("Accuracy is "+str(correct/len(y2)*100)+"%.") # Prints out accuracy