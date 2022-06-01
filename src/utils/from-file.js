const Neural = require('../classifiers/neural');
const Genetic = require('../classifiers/genetic');
const IllegalArgumentException = require('../exceptions/illegal-argument');
const { isBrowser } = require('../utils');
const Exception = require('../exceptions/exception');

module.exports = (() => {
    const fs = isBrowser() ? {} : require('fs');

    const CLASSIFIERS = {
        [Neural.name]: Neural,
        [Genetic.name]: Genetic
    };

    return isBrowser() ?
        async (file) => {
            if (!(file instanceof Blob)) throw new IllegalArgumentException('`file` must be an instance of File');

            return new Promise((resolve, reject) => {
                let reader = new FileReader();
                reader.onload = () => {
                    const model = JSON.parse(reader.result);
                    const classifier = new CLASSIFIERS[model.name]().loadModel(model);

                    resolve(classifier);
                };
                reader.onerror = () => reject(new IllegalArgumentException('`file` must be an instance of File'));
                reader.readAsText(file);
            });
        } :
        async (path) => {
            return new Promise((resolve, reject) => {
                fs.readFile(path, (error, data) => {
                    if (error) return reject(new Exception('No such file'));
                    const model = JSON.parse(data);
                    const classifier = new CLASSIFIERS[model.name]().loadModel(model);

                    resolve(classifier);
                });
            });

        };
})();