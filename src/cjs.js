const fs = require('fs');
const GeneticFX = require('./classifiers/genetic');
const NeuralFX = require('./classifiers/neural');
const Exception = require('./exceptions/exception');

const classifiers = {
    [NeuralFX.name]: NeuralFX,
    [GeneticFX.name]: GeneticFX
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
    NeuralFX,
    GeneticFX,
    fromFile
};