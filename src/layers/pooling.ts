import { Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import { calculatePooledMatrix } from "../lib/utils";
import Layer from "./layer";

export type PoolingParams = {
    input: [number, number];
    window: [number, number]; // just allow for one value?? (symmetric)
    /**
     * @default window width
     */
    stride?: number;
};

export default abstract class PoolingLayer extends Layer {

    window: [number, number];
    stride: number;

    constructor({
        input,
        window,
        stride
    }: PoolingParams) {
        stride = stride || window[0]; // only works for hor/ver symmetry
        const output = calculatePooledMatrix(...input, window[0], stride, 0);

        super(input, output, new Sigmoid());

        this.stride = stride;
        this.window = window;
    }

    // rename maybe and make more elegant!
    abstract backPropagatePoolIndex(aggregate: number, value: number, index: number, indices: number[]): number;

    backPropagate(input: Matrix, _: any, loss: Matrix) {
        const gradient = new Matrix(...this.input);

        for (let i = 0; i < loss.rows; i++) {
            for (let j = 0; j < loss.columns; j++) {

                let aggregate = -Number.MAX_VALUE, // move to only be included in max pooling?
                    indices: number[] = [];

                for (let k = 0; k < this.window[0]; k++) {
                    for (let l = 0; l < this.window[1]; l++) {
                        const index = (i * this.stride + k) * this.input[1] + (j * this.stride + l);
                        if (index >= input.entries.length) continue;

                        aggregate = this.backPropagatePoolIndex(aggregate, input.entries[index], index, indices);
                    }
                }

                for (const index of indices) {
                    gradient.entries[index] += loss.entries[i * loss.columns + j] / indices.length;
                }
            }
        }

        return gradient;
    }

}