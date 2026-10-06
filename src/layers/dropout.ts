import { Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import Layer from "./layer";

export type DropoutParams = {
    input: number | [number, number];
    /**
     * @default 0.25
     */
    rate?: number;
};

export default class DropoutLayer extends Layer {

    name = 'drop';
    rate: number;
    private mask?: Matrix;

    constructor({
        input,
        rate = 0.25
    }: DropoutParams) {
        if (!Array.isArray(input)) input = [input, 1];
        super(input, input, new Sigmoid());

        this.rate = rate;
    }

    propagate(input: Matrix) {
        input.reshape(...this.input);

        this.mask = new Matrix(input.rows, input.columns);

        for (let i = 0; i < this.mask.entries.length; i++) {
            this.mask.entries[i] = Math.random() < this.rate ? 0 : 1 / (1 - this.rate);
        }

        return new Matrix(input).scale(this.mask);
    }

    backPropagate(_1: Matrix, _2: Matrix, loss: Matrix) {
        loss.reshape(...this.output);
        
        if (this.mask) {
            return new Matrix(loss).scale(this.mask);
        }

        return new Matrix(loss);
    }

}