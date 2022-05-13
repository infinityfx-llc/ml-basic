# ml-basic

[![NPM package](https://img.shields.io/npm/v/ml-basic)](https://www.npmjs.com/package/ml-basic)
![NPM bundle size](https://img.shields.io/bundlephobia/minzip/ml-basic)
![NPM weekly downloads](https://img.shields.io/npm/dw/ml-basic)

Lightweight, zero dependency, machine learning library for use in NodeJS and browsers.

## Get started

### Installation
Install the package using NPM, run:

```sh
$ npm i ml-basic
```

## Usage

### NodeJS
```javascript
const MLBasic = require('ml-basic');

// creates a fully connected neural classifier with 2 inputs neurons, 2 hidden neurons and 1 output neuron

const classifier = new MLBasic.classifiers.neural({
    shape: [2, 3, 1]
});

const result = await classifier.predict([1, 0]);

// result = [0.532..]
```

### Browser
When using the package browser-side, you can either include it as a script or import the package when you have it installed locally.
```html
<script src="https://unpkg.com/browse/ml-basic/src/main.js" type="text/javascript"></script>
```

```javascript
import MLBasic from 'ml-basic';
```

```javascript
// creates a fully connected neural classifier with 2 inputs neurons, 2 hidden neurons and 1 output neuron

const classifier = new MLBasic.classifiers.neural({
    shape: [2, 3, 1]
});

const result = await classifier.predict([1, 0]);

// result = [0.532..]
```