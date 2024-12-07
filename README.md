# ml-basic

[![NPM package](https://img.shields.io/npm/v/ml-basic)](https://www.npmjs.com/package/ml-basic)
![NPM bundle size](https://img.shields.io/bundlephobia/minzip/ml-basic)
![Last commit](https://img.shields.io/github/last-commit/infinityfx-llc/ml-basic)
![NPM weekly downloads](https://img.shields.io/npm/dw/ml-basic)
![NPM downloads](https://img.shields.io/npm/dt/ml-basic)

Lightweight, zero dependency, machine learning library for use in NodeJS and browsers.

## Table of contents
- [Importing](#importing)
- [Usage](#usage)
- [Training](#training)

## Importing

### CommonJS
```javascript
const MLBasic = require('ml-basic');
```

### ESM
```javascript
import MLBasic from 'ml-basic';
```

### Browser
```html
<script src="https://unpkg.com/browse/ml-basic/index.js" type="text/javascript"></script>
```

## Usage

```javascript
import { Neural } from 'ml-basic';

const net = new Neural({
    layers: [
        ...
    ]
});

const result = net.predict([1, 0]);

// result = [0.532..]
```

## Training

```javascript
import { DataFrame } from 'ml-basic';

const data = new DataFrame([ ... ]);

const error = await net.fit(data);

// error = 0.532..
```