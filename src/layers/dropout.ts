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

    constructor({
        input,
        rate = 0.25
    }: DropoutParams) {
        if (!Array.isArray(input)) input = [input, 1];
        super(input, input, new Sigmoid());

        this.rate = rate;
    }

    propagate(input: Matrix) {
        return new Matrix(input.rows, input.columns, new Array(this.input[0] * this.input[1]).fill(0).map((_, i) => {
            return Math.random() < this.rate ? 0 : input.entries[i];
        }));
    }

    backPropagate(_1: Matrix, _2: Matrix, loss: Matrix) {
        return loss;
    }

}