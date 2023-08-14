# Softmax-Activation
Provide a way to interpret the models outputs as probabilities across multiple classes

This works by exponentiating each logit (unnormalized predictions) and then normalizing them by dividing by the sum of all the exponential logits into probabilites that sum to 1  

This is applied to the outer layer of the neural network with multiple classes

An Example:  
  - In image classification. If you train a neural network on image classification with images classified in different categories the final layer of tthe neural network uses softmax to convert the score into probabilites allowing you to determine the probabilitiy of the image belonging to the class
