const Exception = require('../exceptions/exception');
const IllegalArgumentException = require('../exceptions/illegal-argument');
const Activator = require('../functions/activator');
const Sigmoid = require('../functions/sigmoid');
const Matrix = require('../math/matrix');
const BatchGradientDescent = require('../optimizers/batch-gradient-descent');
const Optimizer = require('../optimizers/optimizer');
const { TYPES } = require('../types');
const { pad } = require('../utils');

module.exports = class Layer {

    static name = 'identity';

    constructor(activation = Sigmoid, optimizer = BatchGradientDescent, hyper_parameters = {}) {
        if (this.constructor === Layer) throw new Exception('Cannot instantiate abstract class Layer');

        if (!(activation.prototype instanceof Activator)) throw new IllegalArgumentException('`activation` must be of type Activator');
        if (!(optimizer === Optimizer || optimizer.prototype instanceof Optimizer)) throw new IllegalArgumentException('`optimizer` must be of type Optimizer');

        this.activation = activation;
        this.optimizer = new optimizer(hyper_parameters);
        this.hyper_parameters = hyper_parameters;

        this.outputs = [];
        this.inputs = [];
    }

    clone(layer = new Layer()) {
        layer.shape = this.shape;
        layer.activation = this.activation;
        layer.optimizer = new this.optimizer.__proto__.constructor(this.hyper_parameters);

        return layer;
    }

    equals(layer) {
        return Object.entries(this.shape).reduce((bool, [key, val]) => bool && val === layer[key], true) &&
            this.activation === layer.activation &&
            this.optimizer.__proto__.constructor === layer.optimizer.__proto__.constructor;
    }

    flush() {
        this.optimizer.flush();
        this.outputs = [];
        this.inputs = [];
    }

    shapeData(data, shape) {
        if (!Array.isArray(data) && !(data instanceof Matrix)) throw new IllegalArgumentException('`input` must be an instance of Array or Matrix');
        shape = Array.isArray(shape) ? shape.length < 2 ? [shape, 1] : shape : [shape, 1];
        if (Array.isArray(data) && shape[1] === 1) data = pad(data, shape[0]);

        if (!(data instanceof Matrix)) data = Matrix.fromArray(data);
        if (data.entries.length !== shape[0] * shape[1]) throw new IllegalArgumentException('`input` size must match layer input size');

        return data.reshape(...shape);
    }

    propagate(input, f) {
        if (!Array.isArray(input) || !Array.isArray(input[0])) input = [input];

        this.inputs = [];
        this.outputs = input.map(input => f(input));
        return this.outputs;
    }

    backPropagate(loss, f) {
        if (!this.outputs.length || !this.inputs.length) throw new Exception();

        if (!Array.isArray(loss) || !Array.isArray(loss[0])) loss = [loss];

        return loss.map((loss, i) => f(this.inputs[i], this.outputs[i], loss)); // use bind() to get this argument
    }

    static cross(a, b) {
        if (!(a instanceof Layer && a.__proto__.constructor === b.__proto__.constructor)) throw new IllegalArgumentException('`a` and `b` must be instances of Layer');
        if (!a.equals(b)) throw new IllegalArgumentException('`a` must be of equal type to `b`');

        return a.clone().cross(b);
    }

    serialize() {
        return Object.entries(this).reduce((map, [key, val]) => {
            if (val.prototype instanceof Activator || val instanceof Optimizer || val instanceof Matrix) val = val.serialize();

            return { ...map, [key]: val };
        }, { name: this.__proto__.constructor.name });
    }

    static deserialize(data, layer = new Layer()) {
        Object.entries(data).forEach(([key, val]) => {
            if (key === 'activation') return layer[key] = TYPES[val];
            if (key === 'optimizer') return layer[key] = TYPES[val.name].deserialize(val);
            // if (key === 'weights' || key === 'bias' || key === 'kernel') return layer[key] = Matrix.deserialize(val);
            if ('rows' in val && 'columns' in val && 'entries' in val) return layer[key] = Matrix.deserialize(val);

            layer[key] = val;
        });

        return layer;
    }

}