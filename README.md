# ml-basic

[![NPM package](https://img.shields.io/npm/v/ml-basic)](https://www.npmjs.com/package/ml-basic)
![NPM bundle size](https://img.shields.io/bundlephobia/minzip/ml-basic)
![NPM weekly downloads](https://img.shields.io/npm/dw/ml-basic)

Lightweight machine learning library for use in NodeJS and browsers.

## Get started

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
```html
<script src="https://unpkg.com/browse/ml-basic/src/main.js" type="text/javascript"></script>
```

```javascript
// creates a fully connected neural classifier with 2 inputs neurons, 2 hidden neurons and 1 output neuron

const classifier = new MLBasic.classifiers.neural({
    shape: [2, 3, 1]
});

const result = await classifier.predict([1, 0]);

// result = [0.532..]
```