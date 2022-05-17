const Genetic = require('./classifiers/genetic');
const Neural = require('./classifiers/neural');
const PreProcessor = require('./pre-processor');
const fromFile = require('./utils/from-file');

module.exports = {
    PreProcessor,
    Neural,
    Genetic,
    fromFile
};