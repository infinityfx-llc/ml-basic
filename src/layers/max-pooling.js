const Sigmoid = require('../functions/sigmoid');
const Matrix = require('../math/matrix');
const Layer = require('./layer');

module.exports = class MaxPoolingLayer extends Layer {

    static name = 'max_pooling';

    constructor({ input = [8, 8], size = [2, 2], stride = 2, activation = Sigmoid } = {}) {
        super(activation);

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

    shapeInput(input) {
        return this.shapeData(input, this.shape.input);
    }

    propagate(input) {
        input = this.shapeInput(input);

        const rows = (input.rows - this.size[0]) / this.stride + 1,
            cols = (input.columns - this.size[1]) / this.stride + 1;
        const result = new Matrix(...this.shape.output);

        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.columns; j++) {

                let max = - Number.MAX_VALUE;
                for (let k = 0; k < this.size[0]; k++) {
                    for (let l = 0; l < this.size[1]; l++) {
                        const rowIdx = i > rows ? i - 1 : i,
                            colIdx = j > cols ? j - 1 : j;

                        let index = (rowIdx + k) * input.columns + (colIdx * this.stride + l);
                        max = input.entries[index] > max ? input.entries[index] : max;
                    }
                }

                result.entries[i * result.columns + j] = max;
            }
        }

        return result;//.transform(this.activation.function);
    }

    backPropagate(input, _, loss) {
        input = Matrix.reshape(input, ...this.shape.input);
        loss.reshape(...this.shape.output);

        const rows = (input.rows - this.size[0]) / this.stride + 1,
            cols = (input.columns - this.size[1]) / this.stride + 1;
        const result = new Matrix(input.rows, input.columns);

        for (let i = 0; i < loss.rows; i++) {
            for (let j = 0; j < loss.columns; j++) {

                let max = - Number.MAX_VALUE;
                let maxIdx = 0;

                for (let k = 0; k < this.size[0]; k++) {
                    for (let l = 0; l < this.size[1]; l++) {
                        const rowIdx = i > rows ? i - 1 : i,
                            colIdx = j > cols ? j - 1 : j;

                        let index = (rowIdx + k) * input.columns + (colIdx * this.stride + l);
                        max = input.entries[index] > max ? (maxIdx = index, input.entries[index]) : max;
                    }
                }

                result[maxIdx] = loss[i * loss.columns + j];
            }
        }

        return result;
    }

    cross() {
        return this;
    }

    mutate() {
        return this;
    }

    static deserialize(data) {
        return super.deserialize(data, new MaxPoolingLayer());
    }

}