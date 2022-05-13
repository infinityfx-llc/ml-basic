const GeneticFX = require('./classifiers/genetic');
const NeuralFX = require('./classifiers/neural');
const IllegalArgumentException = require('./exceptions/illegal-argument');

const classifiers = {
    [NeuralFX.name]: NeuralFX,
    [GeneticFX.name]: GeneticFX
};

const fromFile = async (file) => {
    if (!(file instanceof Blob)) throw new IllegalArgumentException('File must be an instance of File');

    return new Promise((resolve, reject) => {
        let reader = new FileReader();
        reader.onload = () => {
            const model = JSON.parse(reader.result);
            const classifier = new classifiers[model.name]().loadModel(model);

            resolve(classifier);
        };
        reader.onerror = () => reject(new IllegalArgumentException('File must be an instance of File'));
        reader.readAsText(file);
    });
};

module.exports = {
    classifiers,
    NeuralFX,
    GeneticFX,
    fromFile
};