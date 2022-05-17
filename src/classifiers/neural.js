const IllegalArgumentException = require('../exceptions/illegal-argument');
const Loss = require('../functions/loss');
const SquaredLoss = require('../functions/squared-loss');
const Layer = require('../layer');
const BatchGradientDescent = require('../optimizers/batch-gradient-descent');
const Pool = require('../threading/pool');
const Task = require('../threading/task');
const { TYPES } = require('../types');
const { pad, range, shuffle, argmax } = require('../utils');
const Classifier = require('./classifier');

//return log object from fitting

module.exports = class Neural extends Classifier {

    static name = 'neural';

    constructor({ shape = [2, 1], optimizer = BatchGradientDescent.name, loss_function = SquaredLoss.name, hyper_parameters = {}, options = {} } = {}) {
        super(options.multithreaded);

        this.epochs = 0;
        this.error = 1;
        this.loss = TYPES[loss_function];

        [this.layers, this.shape] = this.createNetwork(shape, optimizer, hyper_parameters);

        this.labels = options.binary ? pad(options.labels, this.shape.output) : null;
        this.binary = options.binary || false;
    }

    async propagate(input) {
        return await Classifier.propagate({
            input,
            network: this.layers,
            input_size: this.shape.input
        });
    }

    async predict(input) {
        const outputs = this.multithreading ?
            await this.pool.queue(
                Task.Propagate({
                    input,
                    network: this.layers.map(layer => layer.serialize()),
                    input_size: this.shape.input
                })
            ) :
            await this.propagate(input);
        let output = outputs[outputs.length - 1];
        output = this.multithreading ? Array.prototype.slice.call(output.entries) : output.toArray();

        if (this.binary) {
            const index = argmax(output);

            return {
                label: this.labels[index],
                certainty: output[index]
            };
        }

        return output;
    }

    async backPropagate(input, target, hyper_parameters = {}) {
        return await Neural.backPropagate({
            input,
            target,
            network: this.layers,
            shape: this.shape,
            loss_function: this.loss,
            hyper_parameters
        });
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

                const error = this.multithreading ?
                    await this.pool.queue(Task.BackPropagate({
                        input,
                        target,
                        network: this.layers.map(layer => layer.serialize()),
                        shape: {
                            input: this.shape.input,
                            output: this.shape.output
                        },
                        loss_function: this.loss.serialize(),
                        hyper_parameters
                    })) :
                    await this.backPropagate(input, target, hyper_parameters);
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
        return JSON.stringify(Object.assign({ name: this.__proto__.constructor.name }, this), (_, value) => {
            if (!value) return value;
            if (value.prototype instanceof Loss || value instanceof Layer) return value.serialize();
            if (value instanceof Pool || key === 'multithreading') return;

            return value;
        }, '\t');
    }

};