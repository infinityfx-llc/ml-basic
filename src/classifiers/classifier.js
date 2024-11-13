const Exception = require('../exceptions/exception');
const IllegalArgumentException = require('../exceptions/illegal-argument');
const Loss = require('../functions/loss');
const SquaredLoss = require('../functions/squared-loss');
const Layer = require('../layers/layer');
const LAYERS = require('../layers');
const Log = require('../log');
const Matrix = require('../math/matrix');
const BatchGradientDescent = require('../optimizers/batch-gradient-descent');
const Pool = require('../threading/pool');
const { TYPES } = require('../types');
const { argmin, isBrowser, shuffle, range } = require('../utils');
const RecurrentLayer = require('../layers/recurrent');

module.exports = (() => {

    const os = isBrowser() ? {
        cpus: () => ({
            length: navigator.hardwareConcurrency || 4
        })
    } :
        require('os');
    const fs = isBrowser() ? null : require('fs');

    return class Classifier {

        constructor(multithreaded = false) {
            if (this.constructor === Classifier) throw new Exception('Cannot instantiate abstract class Classifier');

            try {
                this.multithreading = isBrowser() ? typeof self.Worker !== undefined : typeof require.resolve('worker_threads') === 'string';
                if (!multithreaded) this.multithreading = false;
            } catch (err) {
                this.multithreading = false;
            }

            this.pool = this.multithreading ? new Pool(os.cpus().length) : null;
            this.compiled = true;
        }

        parseLayer(shape, input = null) {
            if (!input) return typeof shape !== 'number' && !Array.isArray(shape) ? shape.size : shape;

            let {
                type = 'fully_connected',
                size = 1,
                activation = 'sigmoid',
                ...args
            } = typeof shape !== 'number' && !Array.isArray(shape) ? shape : {};

            if (typeof shape === 'number') {
                type = 'fully_connected';
                size = shape;
            }
            if (Array.isArray(shape)) {
                type = 'convolutional';
                kernel = shape;
            }

            if (type !== 'fully_connected') {
                input = Array.isArray(input) ? input : (n = Math.sqrt(input), [n, n]);
                if (!Number.isInteger(input[0])) throw new IllegalArgumentException('Layer input shape is invalid for convolutional layers');
            } else {
                input = Array.isArray(input) ? input[0] * input[1] : input;
            }

            return [LAYERS[type], { input, output: size, activation: TYPES[activation], size, ...args }];
        }

        isValidNetwork(network) {
            if (!Array.isArray(network)) return false;

            let previous, shape = [];
            for (const layer of network) {
                if (!(layer instanceof Layer)) return false;

                const input = layer instanceof RecurrentLayer ?
                    layer.shape.input[0] :
                    Array.isArray(layer.shape.input) ?
                        layer.shape.input[0] * layer.shape.input[1] :
                        layer.shape.input;

                const output = layer instanceof RecurrentLayer ?
                    layer.shape.output[0] :
                    Array.isArray(layer.shape.output) ?
                        layer.shape.output[0] * layer.shape.output[1] :
                        layer.shape.output;

                if (!previous) {
                    previous = output;
                    shape.push(layer.shape.input);
                    continue;
                }

                if (previous !== input) return false;
                previous = input;
                shape.push(layer.shape.input);
            }
            shape.push(network[network.length - 1].shape.output);
            shape.input = shape[0];
            shape.output = shape[shape.length - 1];

            return shape;
        }

        createNetwork(shape = [2, 1], optimizer = BatchGradientDescent.name, hyper_parameters = {}) {
            if (!Array.isArray(shape) || shape.length < 2) throw new IllegalArgumentException('`shape` must be an instance of Array with at least 2 values');

            const network = new Array(shape.length - 1);
            const __shape = new Array(shape.length);

            let input = this.parseLayer(shape[0]);
            for (let i = 1; i < shape.length; i++) {
                const [layerType, args] = this.parseLayer(shape[i], input);

                network[i - 1] = new layerType({ ...args, optimizer: TYPES[optimizer], hyper_parameters });

                __shape[i - 1] = input;
                __shape[i] = input = network[i - 1].shape.output;
            }

            __shape.input = __shape[0];
            __shape.output = __shape[__shape.length - 1];

            return [network, __shape];
        }

        static async propagate(input, network) {
            let output = input;//network[0].shapeInput(input), outputs = [output];

            for (let i = 0; i < network.length; i++) {
                if (!(network[i] instanceof Layer)) throw new IllegalArgumentException('`network` must be an Array of Layers');

                output = network[i].propagate(output);
                // outputs.push(output);
            }

            // return outputs;
            return output;
        }

        static async backPropagate(input, target, network, { loss_function = SquaredLoss, hyper_parameters = {} } = {}) {
            const output_size = network[network.length - 1].shape.output;
            if (!Array.isArray(target) || target.length !== output_size) throw new IllegalArgumentException(`\`target\` must be an instance of Array of length ${output_size}`);
            if (!(loss_function.prototype instanceof Loss)) throw new IllegalArgumentException('`loss_function` must be an instance of Loss');

            target = Matrix.fromArray(target);
            // const outputs = await this.propagate(input, network);
            // const output = outputs[outputs.length - 1];
            const output = await this.propagate(input, network);

            let error = loss_function.mean(output, target);
            let loss = loss_function.derivative(output, target); // GET BACK ARRAY

            // for (let i = outputs.length - 1; i > 0; i--) {
            for (let i = network.length - 1; i >= 0; i--) {
                // loss = network[i - 1].backPropagate(outputs[i - 1], outputs[i], loss, hyper_parameters);
                loss = network[i].backPropagate(loss, hyper_parameters);
            }

            return [error, network];
        }

        static async fit(data, network, { loss_function = SquaredLoss, options = {} } = {}) {
            const {
                max_epochs = 1,
                error_threshold = 0,
                iterative = false,
                hyper_parameters = {}
            } = options;
            if (!Array.isArray(data)) throw new IllegalArgumentException('`data` must be an instance of Array');

            const log = new Log();

            for (let i = 0; i < max_epochs; i++) {
                const arr = shuffle(range(data.length));

                let aggregate_error = 0;
                for (let j = 0; j < arr.length; j++) {
                    const index = iterative ? j : arr[j];
                    const { input, target } = data[index];

                    if (!input || !target) throw new IllegalArgumentException('`data` entry is missing an `input` or `target` key and value');

                    const [error] = await this.backPropagate(input, target, network, { loss_function, hyper_parameters });
                    aggregate_error += error / arr.length;
                }

                log.increment({ epochs: 1 });
                log.set({ error: aggregate_error });
                log.add({
                    error: aggregate_error,
                    epoch: log.epochs
                });

                if (log.error <= error_threshold) break;
            }

            return [log.end(), network];
        }

        async tune(iteration_function, parameters = {}) {
            const entries = Object.entries(parameters);
            const size = entries.reduce((s, [_, arr]) => {
                if (!Array.isArray(arr)) throw new IllegalArgumentException('`parameters` entry must be an instance of Array');
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

            const index = argmin(validation_matrix);

            return entries.reduce((map, [key, array], i) => {
                i = Math.floor(index / entries.reduce((t, [_, arr], j) => t * (j > i ? arr.length : 1), 1));
                return { ...map, [key]: array[i % array.length] };
            }, {});
        }

        export(path) {
            let data = this.serialize();
            if (!/\.json$/i.test(path)) path = path + '.json';

            if (!isBrowser()) return fs.writeFileSync(path, data);

            data = new Blob([data], { type: 'application/json' });
            let url = URL.createObjectURL(data),
                a = document.createElement('a');
            a.href = url;
            a.download = path.match(/[^\/|\\]*$/)[0];
            a.click();
            URL.revokeObjectURL(data);
        }

        end() {
            if (this.multithreading) this.pool.flush();
        }

    };

})();