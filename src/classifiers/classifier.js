const Exception = require('../exceptions/exception');
const IllegalArgumentException = require('../exceptions/illegal-argument');
const Sigmoid = require('../functions/sigmoid');
const Layer = require('../layer');
const Matrix = require('../math/matrix');
const BatchGradientDescent = require('../optimizers/batch-gradient-descent');
const { TYPES } = require('../types');
const { argmin, pad } = require('../utils');

module.exports = class Classifier {

    constructor() {
        if (this.constructor === Classifier) throw new Exception('Cannot instantiate abstract class Classifier');
    }

    parseShape(value) {
        if (typeof value !== 'number') return [value.size, TYPES[value.activation]];

        return [value, Sigmoid];
    }

    createNetwork(shape = [2, 1], optimizer = BatchGradientDescent.name, hyper_parameters = {}) {
        if (shape.length < 2) throw new IllegalArgumentException('Shape has to be an instance of Array with at least 2 values');

        const network = new Array(shape.length - 1);
        const __shape = new Array(shape.length);

        for (let i = 0; i < shape.length - 1; i++) {
            const [input] = this.parseShape(shape[i]);
            const [output, activation] = this.parseShape(shape[i + 1]);

            network[i] = new Layer(input, output, activation, TYPES[optimizer], hyper_parameters);
            __shape[i] = input;
            __shape[i + 1] = output;
        }

        __shape.input = __shape[0];
        __shape.output = __shape[__shape.length - 1];

        return [network, __shape];
    }

    async predict(network, input) {
        if (!Array.isArray(input)) throw new IllegalArgumentException('Input must be an instance of Array');

        input = pad(input, this.shape.input);
        let output = Matrix.fromArray(input);

        for (let i = 0; i < network.length; i++) output = network[i].propagate(output);

        return output.toArray();
    }

    async tune(iteration_function, parameters = {}) {
        const entries = Object.entries(parameters);
        const size = entries.reduce((s, [_, arr]) => {
            if (!Array.isArray(arr)) throw new IllegalArgumentException('Parameter entry must be an instance of Array');
            return s ? s * arr.length : arr.length;
        }, null);
        const validation_matrix = new Array(size).fill(0);

        for (let i = 0; i < validation_matrix.length; i++) {
            const hyper_parameters = entries.reduce((map, [key, array], index) => {
                index = Math.floor(i / entries.reduce((t, [_, arr], i) => t * (i > index ? arr.length : 1), 1));
                return { ...map, [key]: array[index % array.length] };
            }, {});

            this.flush();

            validation_matrix[i] = await iteration_function(hyper_parameters);
        }

        console.log(validation_matrix);
        const index = argmin(validation_matrix);

        return entries.reduce((map, [key, array], i) => {
            i = Math.floor(index / entries.reduce((t, [_, arr], j) => t * (j > i ? arr.length : 1), 1));
            return { ...map, [key]: array[i % array.length] };
        }, {});
    }

    export(filename) {
        let data = new Blob([this.serialize()], { type: 'application/json' }),
			url = window.URL.createObjectURL(data),
			a = document.createElement("a");
		a.href = url;
		a.download = filename + '.json';
		a.click();
		window.URL.revokeObjectURL(data);
    }

};