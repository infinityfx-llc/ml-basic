const Exception = require('../exceptions/exception');
const IllegalArgumentException = require('../exceptions/illegal-argument');
const Sigmoid = require('../functions/sigmoid');
const Matrix = require('../math/matrix');
const Layer = require('./layer');

module.exports = class PoolingLayer extends Layer {

    static name = 'pooling';

    constructor({ input = [8, 8], size = [2, 2], stride = 2, activation = Sigmoid } = {}) {
        super(activation);
        if (this.constructor === PoolingLayer) throw new Exception('Cannot instantiate abstract class PoolingLayer');

        if (!Array.isArray(input) || input[0] < 1 || input[1] < 1 || !Number.isInteger(input[0]) || !Number.isInteger(input[1])) throw new IllegalArgumentException('`input` must be an Array of integers greater than 0');
        if (!Array.isArray(size) || size[0] < 1 || size[1] < 1 || !Number.isInteger(size[0]) || !Number.isInteger(size[1])) throw new IllegalArgumentException('`size` must be an Array of integers greater than 0');
        if (stride < 1 || !Number.isInteger(stride)) throw new IllegalArgumentException('`stride` must be an integer greater than 0');

        this.shape = {
            input,
            output: [
                Math.ceil((input[0] - size[0]) / stride + 1),
                Math.ceil((input[1] - size[1]) / stride + 1)
            ]
        };
        this.stride = stride;
        this.size = size;
    }

    clone(layer = new PoolingLayer()) {
        layer = super.clone(layer);
        layer.stride = this.stride;
        layer.size = this.size;

        return layer;
    }

    shapeInput(input) {
        return this.shapeData(input, this.shape.input);
    }

    propagate(input) {
        return super.propagate(input, function (input) {
            input = this.shapeInput(input);
            this.inputs.push(input);

            const rows = (input.rows - this.size[0]) / this.stride + 1,
                cols = (input.columns - this.size[1]) / this.stride + 1;
            const result = new Matrix(...this.shape.output);

            for (let i = 0; i < result.rows; i++) {
                for (let j = 0; j < result.columns; j++) {

                    let aggregate = null;
                    for (let k = 0; k < this.size[0]; k++) {
                        for (let l = 0; l < this.size[1]; l++) {
                            const rowIdx = i > rows ? i - 1 : i,
                                colIdx = j > cols ? j - 1 : j;

                            let index = (rowIdx + k) * input.columns + (colIdx * this.stride + l);
                            aggregate = this.pool(input.entries[index], input.entries.length, aggregate);
                        }
                    }

                    result.entries[i * result.columns + j] = aggregate;
                }
            }

            return result;//.transform(this.activation.function);
        }.bind(this));
    }

    backPropagate(input, _, loss) {
        return super.backPropagate(loss, function (input, _, loss) {
            input = Matrix.reshape(input, ...this.shape.input);
            loss.reshape(...this.shape.output);

            const rows = (input.rows - this.size[0]) / this.stride + 1,
                cols = (input.columns - this.size[1]) / this.stride + 1;
            const result = new Matrix(input.rows, input.columns);

            for (let i = 0; i < loss.rows; i++) {
                for (let j = 0; j < loss.columns; j++) {

                    let aggregate, indices;

                    for (let k = 0; k < this.size[0]; k++) {
                        for (let l = 0; l < this.size[1]; l++) {
                            const rowIdx = i > rows ? i - 1 : i,
                                colIdx = j > cols ? j - 1 : j;

                            let index = (rowIdx + k) * input.columns + (colIdx * this.stride + l);
                            [aggregate, indices] = this.unpool(input.entries[index], index, aggregate, indices);
                        }
                    }

                    indices.forEach(index => result[index] = loss[i * loss.columns + j] / indices.length);
                }
            }

            return result;
        }.bind(this));
    }

    cross() {
        return this;
    }

    mutate() {
        return this;
    }

    static deserialize(data, layer = new PoolingLayer()) {
        return super.deserialize(data, layer);
    }

}