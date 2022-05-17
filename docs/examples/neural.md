# Neural classifier example

```javascript
// Creates a fully connected neural classifier with 4 input neurons, 2 hidden layers with 8 and 4 neurons respectively and 2 output neurons.

const neuralClassifier = MLBasic.Neural({
    shape: [4, 8, 4, 2],
    optimizer: 'batch_gradient_descent',
    loss_function: 'squared_loss',
    hyper_parameters: {
        learning_rate: 0.15,
        batch_size: 6 // The batch size used for batch gradient descent.
    },
    options: {
        multithreaded: true // Default value = true, whether to use multithreading for prediction and fitting tasks.
    }
});

// Some training data set, you would want to have more than 2 samples in a real-world scenario

const data = [
    {
        input: [0.245, 0.832, 0.791, 0.241],
        target: [0.284, 0.579]
    },
    {
        input: [0.891, 0.423, 0.523, 0.972],
        target: [0.623, 0.451]
    },
    // ...
];

// Fit the classifier to the training data set

const error = await neuralClassifier.fit(data, {
    max_epochs: 100, // Train the classifer for a maximum of 100 epochs, where 1 epoch consists of all data samples.
    iterative: false // Default value = false, whether to turn-off order randomization of the data samples.
});

// error = 0.238..

// Export the trained model to a JSON file.

neuralClassifier.export('path/to/model.json'); // In a browser this would result in a file download with the name 'model.json'.

```