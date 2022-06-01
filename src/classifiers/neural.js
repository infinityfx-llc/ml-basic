const Loss = require('../functions/loss');
const SquaredLoss = require('../functions/squared-loss');
const Layer = require('../layers/layer');
const LAYERS = require('../layers');
const Matrix = require('../math/matrix');
const BatchGradientDescent = require('../optimizers/batch-gradient-descent');
const Pool = require('../threading/pool');
const Task = require('../threading/task');
const { TYPES } = require('../types');
const { pad, argmax } = require('../utils');
const Classifier = require('./classifier');
const IllegalArgumentException = require('../exceptions/illegal-argument');
const Exception = require('../exceptions/exception');

// flatten layer sizes for classifier shape (NO 2D arrays)

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

    static model() {
        const model = new Neural();
        model.layers = [];
        model.shape = [];
        model.loss = null;
        model.compiled = false;

        return model;
    }

    add(layer) {
        if (!(layer instanceof Layer)) throw new IllegalArgumentException('`layer` must be instance of Layer');

        this.layers.push(layer);

        return this;
    }

    setLoss(loss_function = SquaredLoss) {
        if (!(loss_function.prototype instanceof Loss)) throw new IllegalArgumentException('`loss_function` must be of type Loss');
        this.loss = loss_function;

        return this;
    }

    compile({ binary = false, labels = null } = {}) {
        if (this.compiled) return;

        if (!this.loss) throw new Exception('Unable to compile model using an undefined `loss_function`');
        this.shape = this.isValidNetwork(this.layers);
        if (!this.shape) throw new Exception('Unable to compile model using invalid layer shapes');

        this.labels = binary ? pad(labels, this.shape.output) : null;
        this.binary = binary;

        this.compiled = true;
        return this;
    }

    async propagate(input) {
        return await Classifier.propagate(input, this.layers);
    }

    async predict(input) {
        const outputs = this.multithreading ?
            await this.pool.queue(
                Task.Propagate({
                    input: input instanceof Matrix ? input.serialize() : input,
                    network: this.layers.map(layer => layer.serialize())
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
        const [error, network] = this.multithreading ?
            await this.pool.queue(Task.BackPropagate({
                input: input instanceof Matrix ? input.serialize() : input,
                target,
                network: this.layers.map(layer => layer.serialize()),
                loss_function: this.loss.serialize(),
                hyper_parameters
            })) :
            await Classifier.backPropagate(
                input,
                target,
                this.layers,
                {
                    loss_function: this.loss,
                    hyper_parameters
                }
            );

        if (this.multithreading) this.layers = network.map(layer => LAYERS[layer.name].deserialize(layer));

        return error;
    }

    flush() {
        this.layers.forEach(layer => layer.flush());
        this.epochs = 0;
    }

    async fit(data, { max_epochs = 1, error_threshold = 0, iterative = false, hyper_parameters = {} } = {}) {
        const [log, network] = this.multithreading ?
            await this.pool.queue(Task.Fit({
                data: data.map(({ input, target }) => ({ input: input instanceof Matrix ? input.serialize() : input, target })),
                network: this.layers.map(layer => layer.serialize()),
                loss_function: this.loss.serialize(),
                options: {
                    max_epochs,
                    error_threshold,
                    iterative,
                    hyper_parameters
                }
            })) :
            await Classifier.fit(
                data,
                this.layers,
                {
                    loss_function: this.loss,
                    options: {
                        max_epochs,
                        error_threshold,
                        iterative,
                        hyper_parameters
                    }
                }
            );

        if (this.multithreading) this.layers = network.map(layer => LAYERS[layer.name].deserialize(layer));
        this.error = log.error;
        this.epochs += log.epochs;

        return log;
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
            if (key === 'layers') return this[key] = val.map(layer => LAYERS[layer.name].deserialize(layer));

            this[key] = val;
        });
        this.shape.input = this.shape[0];
        this.shape.output = this.shape[this.shape.length - 1];

        return this;
    }

    serialize() {
        return JSON.stringify(Object.assign({ name: this.__proto__.constructor.name }, this), (key, value) => {
            if (!value) return value;
            if (value.prototype instanceof Loss || value instanceof Layer) return value.serialize();
            if (value instanceof Pool || key === 'multithreading') return;
            if (value instanceof Float64Array) return Array.prototype.slice.call(value);

            return value;
        }, '\t');
    }

};