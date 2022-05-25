const Classifier = require('../classifiers/classifier');
const LAYERS = require('../layers');
const Matrix = require('../math/matrix');
const { TYPES } = require('../types');

module.exports = class Task {

    static async run(name, shared) {
        return await new Task()[name](shared);
    }

    async propagate(shared) {
        const outputs = await Classifier.propagate(
            Array.isArray(shared.input) ? shared.input : Matrix.deserialize(shared.input),
            shared.network.map(layer => LAYERS[layer.name].deserialize(layer))
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
            Array.isArray(shared.input) ? shared.input : Matrix.deserialize(shared.input),
            shared.target,
            shared.network.map(layer => LAYERS[layer.name].deserialize(layer)),
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
            shared.data.map(({ input, target }) => ({ input: Array.isArray(input) ? input : Matrix.deserialize(input), target })),
            shared.network.map(layer => LAYERS[layer.name].deserialize(layer)),
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