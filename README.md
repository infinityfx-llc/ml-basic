# ml-basic

[![NPM package](https://img.shields.io/npm/v/ml-basic)](https://www.npmjs.com/package/ml-basic)
![NPM bundle size](https://img.shields.io/bundlephobia/minzip/ml-basic)
![Last commit](https://img.shields.io/github/last-commit/infinityfx-llc/ml-basic)
![NPM weekly downloads](https://img.shields.io/npm/dw/ml-basic)
![NPM downloads](https://img.shields.io/npm/dt/ml-basic)

Lightweight, zero dependency, machine learning library for use in NodeJS and browsers.

## Table of contents
- [Get started](#get-started)
    - [Installation](#installation)
    - [Usage](#usage)
- [Prediction](#prediction)
- [Training](#training)
- [Classes](#classes)
    - [Classifiers](#classifiers)
    - [Layers](#layers)
    - [Optimizers](#optimizers)
- [Functions](#functions)
    - [Activation](#activation)
    - [Loss](#loss)
- [Data](#data)
    - [Format](#format)
    - [Pre processing](#pre-processing)
- [Examples](#examples)

## Get started

### Installation
Install the package using NPM, run:

```sh
$ npm i ml-basic
```

### Usage

### NodeJS
```javascript
const MLBasic = require('ml-basic');

// creates a fully connected neural classifier with 2 inputs neurons, 3 hidden neurons and 1 output neuron.

const classifier = new MLBasic.Neural({
    shape: [2, 3, 1]
});

const result = await classifier.predict([1, 0]);

// result = [0.532..]
```

### Browser
When using the package browser-side, you can either include it as a script or import the package when you have it installed locally.
```html
<script src="https://unpkg.com/browse/ml-basic/index.js" type="text/javascript"></script>
```

```javascript
import MLBasic from 'ml-basic';
```

```javascript
// creates a fully connected neural classifier with 2 inputs neurons, 3 hidden neurons and 1 output neuron.

const classifier = new MLBasic.Neural({
    shape: [2, 3, 1]
});

const result = await classifier.predict([1, 0]);

// result = [0.532..]
```

## Prediction

```javascript
const result = await classifier.predict([1, 0]);

// result = [0.532..]
```

## Training

```javascript
const data = [ ... ];

const log = await classifier.fit(data);

// log.error = 0.532..
```

## Classes

### Classifiers

#### Neural classifier
Most common type of classifier which operates on a single network of layers. The classifier converges using back propagation through the network, based on training data.

```javascript
// All arguments listed below are optional

const neuralClassifer = new MLBasic.Neural({
    shape = [ ... ],
    optimizer = 'gradient_descent' | 'batch_gradient_descent' | 'rms_prop' | 'adam',
    loss_function = 'sigmoid' | 'tanh' | 'parameteric_relu' | 'elu' | 'softplus',
    hyper_parameters = { ... }, 
    options = { ... }
});
```

#### Genetic classifier
Classifier based on genetic evolution. The classifier mantains a population of networks which evolve and mutate over time. Convergance is reached based on user defined fitness function which chooses which candidates get to  cross to form a new generation.

```javascript
const geneticClassifier = new MLBasic.Genetic({ ... });
```

#### Classifier options
Optional classifier options which can be passed when creating a classifier, default values are as listed below.
```javascript
{
    multithreaded: false,
    binary: false, // Output a binary result when calling predict
    labels: [ ... ] // An array of strings used to label the outputs when binary is set to true
}
```

### Layers

You are able to define the layers used in your network during initialization of your classifier, using the `shape` argument, which takes an array of values which define each layer type.

```javascript
const classifier = new MLBasic.Neural({
    shape: [ ... ]
});
```

The shape argument can take various layer type definitions that are each parsed as a specific layer type.`

```javascript
const shape = [
    8 | [8, 8], // The first array element defines the input size and can be either a number or 2-value array of numbers.
    5, // Any subsequent number is parsed as a fully connected layer with it's input size defined as the respective number.
    [5, 5], // Any subsequent layer's defined as arrays will be parsed as convolutional layers, where the array values are the layer's kernel size.
    { // Lastly, you are able to define a layer using an object. All possible arguments per layer type are described below under each respective layer's subsection.
        type: 'fully_connected',
        size: 5
    }
];
```

- #### Fully connected layer
    Most common layer type, which maps all input neurons to all output neurons.

    ```javascript
    const layer = new MLBasic.layers.neural({ ... });`
    ```

    Classifier shape definition, optional values default to displayed value:
    ```javascript
    {
        type: 'fully_connected',
        size: 2, // Optional
        activation: 'sigmoid' // Optional
    }
    ```

- #### Convolutional layer
    Uses convolution with a filter/kernel to map input data, usually 2-dimensional data such as images to it's output.

    ```javascript
    const layer = new MLBasic.layers.convolutional({ ... });`
    ```

    Classifier shape definition, optional values default to displayed value:
    ```javascript
    {
        type: 'convolutional',
        kernel: [3, 3], // Optional
        activation: 'sigmoid', // Optional
        stride: 1 // Optional
    }
    ```

- #### Max pooling layer
    Layer which groups input data, based on the maximum value out of the input group.

    ```javascript
    const layer = new MLBasic.layers.max_pooling({ ... });`
    ```

    Classifier shape definition, optional values default to displayed value:
    ```javascript
    {
        type: 'max_pooling',
        size: [2, 2], // Optional
        activation: 'sigmoid', // Optional
        stride: 2 // Optional
    }
    ```

- #### Average pooling layer
    Layer which groups input data, based on the average value out of the input group.

    ```javascript
    const layer = new MLBasic.layers.average_pooling({ ... });`
    ```

    Classifier shape definition, optional values default to displayed value:
    ```javascript
    {
        type: 'average_pooling',
        size: [2, 2], // Optional
        activation: 'sigmoid', // Optional
        stride: 2 // Optional
    }
    ```

- #### Recurrent layer
    Coming soon!

### Optimizers

- #### Gradient descent
    Base optimizer class, which does not affect gradient through time and only uses a static learning rate.

    Optional hyper parameters, default values are as displayed:
    ```javascript
    {
        learning_rate: 0.1,
        gradient_clipping: null
    }
    ```

- #### Batch gradient descent
    Extension of normal gradient descent which groups multiple gradients together and only updates weights per gradient batch.

    Optional hyper parameters, default values are as displayed:
    ```javascript
    {
        learning_rate: 0.1,
        gradient_clipping: null,
        batch_size: 8
    }
    ```

- #### Adam
    Optional hyper parameters, default values are as displayed:
    ```javascript
    {
        learning_rate: 0.01,
        gradient_clipping: null,
        batch_size: 4,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8
    }
    ```

- #### RMSProp
    Optional hyper parameters, default values are as displayed:
    ```javascript
    {
        learning_rate: 0.01,
        gradient_clipping: null,
        batch_size: 4,
        beta: 0.9,
        epsilon: 1e-8
    }
    ```

## Functions

### Activation functions
- Sigmoid - `sigmoid`

- TanH - `tanh`

- Parametric ReLu - `parameteric_relu`

- Elu - `elu`

- Softplus - `softplus`

### Loss functions
- Squared Mean Error - `squared_loss`

- Cross entropy - `cross_entropy`

## Data

### Format

#### Input data
All classifiers expect an `Array` of numbers as input data, even if a classifier has a higher dimensional input shape input data must be a 1-dimensional `Array`. The classifier itself takes care of reshaping the data to fit the right dimensions. If you want to use other primitive data types, such as `String`, you are expected to first process this data into a number format.

#### Training data
All classifiers expect an `Array` of objects containing an `input` and `target` key. These `input` and `target` values are expected to be an `Array` of numbers. Again if your data is not in this format a `PreProcessor` can be used to format the data.

```javascript
const data = [
    {
        input: [ ... ],
        target: [ ... ]
    },
    ...
];
```

### Pre processing
Using the `PreProcessor` class you are able to format, clean and normalize training data. The `PreProcessor` allows you to load various data formats, including from a file.

```javascript
const processor = new MLBasic.PreProcessor();
```

The example below ingests raw data that is formatted using only arrays, cleans and normalizes it, before outputting it.

```javascript
const rawData = [
    [
        [1.421, 8.482, 2.589, ...], // Input data
        [0.875, 5.892, 5.978, ...] // Target data
    ],
    ...
];

preProcessor.clean({ 
    nullToZero: true, // nullToZero (default = true)
    removeDuplicates: true, // removeDuplicates (default = true)
    allowVariableData: false // removeDuplicates (default = true)
});

preProcessor.normalize(0, 1); // min (default = 0), max (default = 1)

const data = preProcessor.out();

// data = [
//     {
//         input: [0.168, 1.000, 0.305, ...],
//         target: [0.146, 0.986, 1.000, ...]
//     },
//     ...
// ]
```

## Examples

- ### [Neural classifier](docs/examples/neural.md)