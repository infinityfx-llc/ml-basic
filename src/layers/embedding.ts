import { Initialization, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import { discrete } from "../lib/utils";
import Layer from "./layer";

export type EmbeddingParams = {
    input: number | [number, number];
    output: number;
    vocabulary: number;
    /**
     * @default 'uniform'
     */
    initialization?: Initialization;
};

export default class EmbeddingLayer extends Layer {

    name = 'embd';
    vocab: number;
    weights: Matrix;

    constructor({
        input = 1,
        output = 0,
        vocabulary = 0,
        initialization = 'uniform'
    }: EmbeddingParams) {
        const seqLen = Array.isArray(input) ? input[0] * input[1] : input;
        super([seqLen, 1], [seqLen, output], new Sigmoid(), initialization);

        this.vocab = vocabulary;
        this.weights = this.initializer(vocabulary, output);
    }

    propagate(input: Matrix) {
        input.reshape(...this.input);

        const output = new Matrix(this.input[0], this.output[1]);

        for (let i = 0; i < this.input[0]; i++) {
            const token = discrete(input.entries[i], 0, this.vocab - 1);

            output.setRow(i, this.weights.row(token));
        }

        return output.reshape(...this.output);
    }

    backPropagate(input: Matrix, _: Matrix, loss: Matrix) {
        input.reshape(...this.input);
        loss.reshape(this.input[0], this.output[1]);

        const gradient = new Matrix(this.vocab, this.output[1]);

        for (let i = 0; i < this.input[0]; i++) {
            const token = discrete(input.entries[i], 0, this.vocab - 1);

            gradient.addRow(token, loss.row(i));
        }

        this.optimizer.tune(this.weights, gradient);
        this.optimizer.flush();

        return new Matrix(this.input[0], this.input[1]);
    }

}
