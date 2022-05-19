const Classifier = require('../classifiers/classifier');
const Layer = require('../layer');
const { TYPES } = require('../types');

module.exports = class Task {

    static async run(name, shared) {
        return await new Task()[name](shared);
    }

    async propagate(shared) {
        const outputs = await Classifier.propagate({
            input: shared.input,
            network: shared.network.map(layer => Layer.deserialize(layer)),
            input_size: shared.input_size
        });

        return outputs.map(matrix => matrix.serialize());
    }

    static Propagate(data) {
        return {
            name: 'propagate',
            shared: data
        };
    }

    async back_propagate(shared) {
        const [error, network] = await Classifier.backPropagate({
            input: shared.input,
            target: shared.target,
            network: shared.network.map(layer => Layer.deserialize(layer)),
            shape: shared.shape,
            loss_function: TYPES[shared.loss_function],
            hyper_parameters: shared.hyper_parameters
        });

        return [error, network.map(layer => layer.serialize())];
    }

    static BackPropagate(data) {
        return {
            name: 'back_propagate',
            shared: data
        };
    }

};