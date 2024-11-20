import { Activator, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import Layer from "./layer";

export default abstract class PoolingLayer extends Layer {

    window: [number, number];
    stride: number;

    constructor({
        input,
        window,
        activation = new Sigmoid()
    }: {
        input: [number, number];
        window: [number, number];
        activation?: Activator;
    }) {
        const output: [number, number] = [
            (input[0] - window[0]) / window[0] + 1, // account for decimal
            (input[1] - window[1]) / window[0] + 1
        ];

        super(input, output, activation);
        this.window = window;
        this.stride = window[0]; // only works for hor/ver symmetry
    }

    backPropagate(input: Matrix, output: Matrix, loss: Matrix) { // todo
        const t = new Matrix(...this.input);

        for (let i = 0; i < t.rows; i++) { // divide by window size
            for (let j = 0; j < t.columns; j++) {

                for (let k = 0; k < this.window[0]; k++) {
                    for (let l = 0; l < this.window[1]; l++) {
                        const index = (i * this.window[0] + k) * loss.columns + (j * this.window[1] + l);

                        const val = loss.entries[index] || 0;
                        // average or find maximum
                    }
                }

                // second loop
                // fill averages or set maximum in correct position
            }
        }

        // todo
        return new Matrix(...this.input);
    }

}