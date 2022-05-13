const IllegalArgumentException = require('../exceptions/illegal-argument');
const Loss = require('../functions/loss');
const SquaredLoss = require('../functions/squared-loss');
const Layer = require('../layer');
const Matrix = require('../math/matrix');
const BatchGradientDescent = require('../optimizers/batch-gradient-descent');
const { TYPES } = require('../types');
const { pad, range, shuffle } = require('../utils');
const Classifier = require('./classifier');

//implement labeled data
//implement binary selection

module.exports = class NeuralFX extends Classifier {

    static name = 'neural';

    constructor({ shape = [2, 1], optimizer = BatchGradientDescent.name, loss_function = SquaredLoss.name, hyper_parameters = {} } = {}) {
        super();

        this.epochs = 0;
        this.error = 1;
        this.loss = TYPES[loss_function];

        [this.layers, this.shape] = this.createNetwork(shape, optimizer, hyper_parameters);
    }

    async predict(input) {
        return await super.predict(this.layers, input);
    }

    async backPropagate(input, target, hyper_parameters) {
        if (!Array.isArray(input)) throw new IllegalArgumentException('Input must be an instance of Array');
        if (!Array.isArray(target) || target.length !== this.shape.output) throw new IllegalArgumentException(`Target must be an instance of Array of length ${this.shape.output}`);

        input = pad(input, this.shape.input);
        let output = Matrix.fromArray(input);
        target = Matrix.fromArray(target);

        const outputs = [output];
        for (let i = 0; i < this.layers.length; i++) {
            output = this.layers[i].propagate(output);
            outputs.push(output);
        }

        let error = this.loss.mean(output, target);
        let loss = this.loss.derivative(output, target);

        for (let i = outputs.length - 1; i > 0; i--) {
            loss = this.layers[i - 1].backPropagate(outputs[i - 1], outputs[i], loss, hyper_parameters);
        }

        return error;
    }

    flush() {
        this.layers.forEach(layer => layer.flush());
        this.epochs = 0;
    }

    async fit(data, { max_epochs = 1, error_threshold = 0, iterative = false, hyper_parameters = {} } = {}) {
        if (!Array.isArray(data)) throw new IllegalArgumentException('Data must be an instance of Array');

        for (let i = 0; i < max_epochs; i++) {
            const arr = shuffle(range(data.length));

            let aggregate_error = 0;
            for (let j = 0; j < arr.length; j++) {
                const index = iterative ? j : arr[j];
                const { input, target } = data[index];

                if (!input || !target) throw new IllegalArgumentException('Data entry must be an Object containing input and target values');

                const error = await this.backPropagate(input, target, hyper_parameters);
                aggregate_error += error / arr.length;
            }

            this.error = aggregate_error;
            this.epochs++;

            if (this.error <= error_threshold) break;
        }

        return this.error;
    }

    async tune(data, parameters = {}, { max_epochs = 1000 } = {}) {
        return await super.tune(async (hyper_parameters) => {
            const error = await this.fit(data, {
                max_epochs,
                hyper_parameters
            });

            return error;
        }, parameters);
    }

    loadModel(model) {
        Object.entries(model).forEach(([key, val]) => {
            if (key === 'loss') return this[key] = TYPES[val];
            if (key === 'layers') return this[key] = val.map(layer => Layer.deserialize(layer));

            this[key] = val;
        });

        return this;
    }

    serialize() {
        return JSON.stringify(Object.assign({ name: this.__proto__.constructor.name } , this), (_, value) => {
            if (!value) return value;
			if (value.prototype instanceof Loss || value instanceof Layer) return value.serialize();

			return value;
		}, '\t');
    }

};