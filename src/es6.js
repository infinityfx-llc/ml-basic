const Genetic = require('./classifiers/genetic');
const Neural = require('./classifiers/neural');
const PreProcessor = require('./pre-processor');
const IllegalArgumentException = require('./exceptions/illegal-argument');

const classifiers = {
    [Neural.name]: Neural,
    [Genetic.name]: Genetic
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
    PreProcessor,
    Neural,
    Genetic,
    fromFile
};