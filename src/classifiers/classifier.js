const Exception = require('../exceptions/exception');
const IllegalArgumentException = require('../exceptions/illegal-argument');
const Loss = require('../functions/loss');
const Sigmoid = require('../functions/sigmoid');
const SquaredLoss = require('../functions/squared-loss');
const Layer = require('../layer');
const Log = require('../log');
const Matrix = require('../math/matrix');
const BatchGradientDescent = require('../optimizers/batch-gradient-descent');
const Pool = require('../threading/pool');
const { TYPES } = require('../types');
const { argmin, pad, isBrowser, shuffle, range } = require('../utils');

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

        static async propagate(input, network) {
            if (!Array.isArray(input)) throw new IllegalArgumentException('Input must be an instance of Array');
            const input_size = network[0].size.input;

            input = pad(input, input_size);
            let output = Matrix.fromArray(input), outputs = [output];

            for (let i = 0; i < network.length; i++) {
                if (!(network[i] instanceof Layer)) throw new IllegalArgumentException('Network must be an Array of Layers');

                output = network[i].propagate(output);
                outputs.push(output);
            }

            return outputs;
        }

        static async backPropagate(input, target, network, { loss_function = SquaredLoss, hyper_parameters = {} } = {}) {
            const output_size = network[network.length - 1].size.output;
            if (!Array.isArray(target) || target.length !== output_size) throw new IllegalArgumentException(`Target must be an instance of Array of length ${output_size}`);
            if (!(loss_function.prototype instanceof Loss)) throw new IllegalArgumentException('Loss function must be an instance of Loss');

            target = Matrix.fromArray(target);
            const outputs = await this.propagate(input, network);
            const output = outputs[outputs.length - 1];

            let error = loss_function.mean(output, target);
            let loss = loss_function.derivative(output, target);

            for (let i = outputs.length - 1; i > 0; i--) {
                loss = network[i - 1].backPropagate(outputs[i - 1], outputs[i], loss, hyper_parameters);
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
            if (!Array.isArray(data)) throw new IllegalArgumentException('Data must be an instance of Array');

            const log = new Log();

            for (let i = 0; i < max_epochs; i++) {
                const arr = shuffle(range(data.length));

                let aggregate_error = 0;
                for (let j = 0; j < arr.length; j++) {
                    const index = iterative ? j : arr[j];
                    const { input, target } = data[index];

                    if (!input || !target) throw new IllegalArgumentException('Data entry must be an Object containing input and target values');

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