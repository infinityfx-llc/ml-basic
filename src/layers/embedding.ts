import { Initialization, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
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
    outputDim: number;
    weights: Matrix;

    constructor({
        input = 1,
        output = 0,
        vocabulary = 0,
        initialization = 'uniform'
    }: EmbeddingParams) {
        const seqLen = Array.isArray(input) ? input[0] * input[1] : input;
        super([seqLen, 1], [seqLen * output, 1], new Sigmoid(), initialization);

        this.vocab = vocabulary;
        this.outputDim = output;
        this.weights = this.initializer(vocabulary, output);
    }

    propagate(input: Matrix) {
        input.reshape(...this.input);

        if (this.input[0] === 1) {
            const token = Math.max(0, Math.min(Math.round(input.entries[0]), this.vocab - 1));
            return this.weights.row(token).reshape(...this.output);
        }

        const output = new Matrix(this.input[0], this.outputDim);
        for (let s = 0; s < this.input[0]; s++) {
            const token = Math.max(0, Math.min(Math.round(input.entries[s]), this.vocab - 1));
            output.setRow(s, this.weights.row(token));
        }

        return output.reshape(...this.output);
    }

    backPropagate(input: Matrix, _: Matrix, loss: Matrix) {
        input.reshape(...this.input);
        loss.reshape(this.input[0], this.outputDim);

        const dWeights = new Matrix(this.vocab, this.outputDim);

        for (let s = 0; s < this.input[0]; s++) {
            const token = Math.max(0, Math.min(Math.round(input.entries[s]), this.vocab - 1));
            dWeights.addRow(token, loss.row(s));
        }

        this.optimizer.tune(this.weights, dWeights);
        this.optimizer.flush();

        return new Matrix(this.input[0], this.input[1]);
    }

}
