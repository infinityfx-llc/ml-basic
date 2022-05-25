# ml-basic

[![NPM package](https://img.shields.io/npm/v/ml-basic)](https://www.npmjs.com/package/ml-basic)
![NPM bundle size](https://img.shields.io/bundlephobia/minzip/ml-basic)
![NPM weekly downloads](https://img.shields.io/npm/dw/ml-basic)

Lightweight, zero dependency, machine learning library for use in NodeJS and browsers.

## Table of contents
- [Get started](#get-started)
    - [Installation](#installation)
    - [Usage](#usage)
- [Classifiers](#classifiers)
    - [Layers](#layers)
- [Pre processor](#pre-processor)
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

## Classifiers

```javascript
const neuralClassifer = new MLBasic.Neural({ ... });

const geneticClassifier = new MLBasic.Genetic({ ... });
```

### Layers

```javascript
// An example of how to define a classifier shape with mixed layer definitions

const neuralClassifer = new MLBasic.Neural({
    shape: [
        [16, 16], // Input layer, 2d-dimensional data
        { type: 'convolutional', kernel: [3, 3], stride: 1 }, // Convolutional layer
        [3, 3], // Assumed to be convolutional layer, with a kernel of size 3x3 and stride of 1
        { type: 'max_pooling', size: [2, 2], stride: 2 } // Max pooling layer
        { activation: 'elu', size: 3 } // Assumed to be fully connected layer with output size 3
    ]
});

// General layer definition in object form

const layer = {
    type: 'fully_connected' | 'convolutional' | 'max_pooling',
    size: 3, // Can be a number or an array when defining a max pooling layer
    kernel: [3, 3], // Only taken into account when defining a convolutional layer
    stride: 2, // Only taken into account when defining either a convolutional or max pooling layer
    activation: 'sigmoid' | 'tanh' | 'parametric_relu' | 'elu' | 'softplus'
};
```

## Pre processor

```javascript
const rawData = [
    [
        [1.421, 8.482, 2.589, ...], // Input data
        [0.875, 5.892, 5.978, ...] // Target data
    ],
    ...
]

// Create a pre processor to convert, clean and normalize raw data.

const preProcessor = new MLBasic.PreProcessor(rawData);

// Clean the data by removing undefined, inconsistent or redundant values.

preProcessor.clean({ nullToZero: true, removeDuplicates: true }); // nullToZero (default = true), removeDuplicates (default = true)

preProcessor.normalize(0, 1); // min = 0 (default = 0), max = 1 (default = 1)

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