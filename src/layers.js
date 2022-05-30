const FullyConnectedLayer = require('./layers/fully-connected');
const ConvolutionalLayer = require('./layers/convolutional');
const MaxPoolingLayer = require('./layers/max-pooling');
const AveragePooling = require('./layers/average_pooling');

module.exports = {
    [FullyConnectedLayer.name]: FullyConnectedLayer,
    [ConvolutionalLayer.name]: ConvolutionalLayer,
    [MaxPoolingLayer.name]: MaxPoolingLayer,
    [AveragePooling.name]: AveragePooling
};