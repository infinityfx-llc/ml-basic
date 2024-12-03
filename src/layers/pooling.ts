import { Activator, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import { calculatePooledMatrix } from "../lib/utils";
import Layer from "./layer";

export default abstract class PoolingLayer extends Layer {

    window: [number, number];
    stride: number;

    constructor({
        input,
        window,
        stride,
        activation = new Sigmoid()
    }: {
        input: [number, number];
        window: [number, number]; // just allow for one value?? (symmetric)
        stride?: number;
        activation?: Activator;
    }) {
        stride = stride || window[0]; // only works for hor/ver symmetry
        const output = calculatePooledMatrix(...input, window[0], stride, 0);

        super(input, output, activation);

        this.stride = stride;
        this.window = window;
    }

    // rename maybe and make more elegant!
    abstract backPropagatePoolIndex(aggregate: number, value: number, index: number, indices: number[]): number;

    backPropagate(input: Matrix, _: any, loss: Matrix) {
        const gradient = new Matrix(...this.input);

        for (let i = 0; i < gradient.rows; i++) {
            for (let j = 0; j < gradient.columns; j++) {

                let aggregate = -Number.MAX_VALUE,
                    indices: number[] = [];

                for (let k = 0; k < this.window[0]; k++) {
                    for (let l = 0; l < this.window[1]; l++) {
                        const index = i * gradient.columns + j;

                        const value = input.entries[index] || 0;
                        aggregate = this.backPropagatePoolIndex(aggregate, value, index, indices);
                    }
                }

                for (const index of indices) {
                    const lossIndex = Math.floor(i / this.window[0]) * this.output[1] + Math.floor(j / this.window[1]);

                    gradient.entries[index] = loss.entries[lossIndex] / indices.length;
                }
            }
        }

        return gradient;
    }

}