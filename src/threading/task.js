const Classifier = require('../classifiers/classifier');
const Layer = require('../layer');
const { TYPES } = require('../types');

module.exports = class Task {

    static async run(name, shared) {
        return await new Task()[name](shared);
    }

    async propagate(shared) {
        const outputs = await Classifier.propagate(
            shared.input,
            shared.network.map(layer => Layer.deserialize(layer))
        );

        return outputs.map(matrix => matrix.serialize());
    }

    static Propagate(data) {
        return {
            name: 'propagate',
            shared: data
        };
    }

    async back_propagate(shared) {
        const [error, network] = await Classifier.backPropagate(
            shared.input,
            shared.target,
            shared.network.map(layer => Layer.deserialize(layer)),
            {
                loss_function: TYPES[shared.loss_function],
                hyper_parameters: shared.hyper_parameters
            }
        );

        return [error, network.map(layer => layer.serialize())];
    }

    static BackPropagate(data) {
        return {
            name: 'back_propagate',
            shared: data
        };
    }

    async fit(shared) {
        const [log, network] = await Classifier.fit(
            shared.data,
            shared.network.map(layer => Layer.deserialize(layer)),
            {
                loss_function: TYPES[shared.loss_function],
                options: shared.options
            }
        );

        return [log, network.map(layer => layer.serialize())];
    }

    static Fit(data) {
        return {
            name: 'fit',
            shared: data
        };
    }

};