const Genetic = require('./classifiers/genetic');
const Neural = require('./classifiers/neural');
const PreProcessor = require('./pre-processor');
const fromFile = require('./utils/from-file');
const layers = require('./layers');
const functions = require('./functions');

module.exports = {
    PreProcessor,
    Neural,
    Genetic,
    fromFile,
    layers,
    functions
};