const FullyConnectedLayer = require('./layers/fully-connected');
const ConvolutionalLayer = require('./layers/convolutional');
const MaxPoolingLayer = require('./layers/max-pooling');

module.exports = {
    [FullyConnectedLayer.name]: FullyConnectedLayer,
    [ConvolutionalLayer.name]: ConvolutionalLayer,
    [MaxPoolingLayer.name]: MaxPoolingLayer
};