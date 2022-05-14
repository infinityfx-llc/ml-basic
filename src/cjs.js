const fs = require('fs');
const Genetic = require('./classifiers/genetic');
const Neural = require('./classifiers/neural');
const PreProcessor = require('./pre-processor');
const Exception = require('./exceptions/exception');

const classifiers = {
    [Neural.name]: Neural,
    [Genetic.name]: Genetic
};

const fromFile = async (path) => {
    return new Promise((resolve, reject) => {
        fs.readFile(path, (error, data) => {
            if (error) return reject(new Exception('No such file'));
            const model = JSON.parse(data);
            const classifier = new classifiers[model.name]().loadModel(model);

            resolve(classifier);
        });
    });

};

module.exports = {
    classifiers,
    PreProcessor,
    Neural,
    Genetic,
    fromFile
};