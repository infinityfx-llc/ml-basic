import Matrix from "../lib/matrix";
import BatchGradientDescent from "./batch-gradient-descent";

export type AdamParams = {
    /**
     * @default 0.01
     */
    learningRate?: number;
    /**
     * @default 0
     */
    clipping?: number;
    /**
     * @default 4
     */
    batchSize?: number;
    /**
     * @default 0.9
     */
    beta1?: number;
    /**
     * @default 0.999
     */
    beta2?: number;
    /**
     * @default 1e-8
     */
    epsilon?: number;
};

export default class Adam extends BatchGradientDescent {

    name = 'adam';
    beta1: number;
    beta2: number;
    epsilon: number;
    private m?: Matrix;
    private v?: Matrix;

    constructor({
        learningRate = 0.01,
        clipping = 0,
        batchSize = 4,
        beta1 = 0.9,
        beta2 = 0.999,
        epsilon = 1e-8
    }: AdamParams = {}) {
        super({ learningRate, clipping, batchSize });

        this.beta1 = beta1;
        this.beta2 = beta2;
        this.epsilon = epsilon;
    }

    step(input: Matrix, gradient: Matrix, callback: (input: Matrix, gradient: Matrix) => void) {
        super.step(input, gradient, (input, gradient) => {
            if (!this.m || !this.v) {
                this.m = new Matrix(gradient.rows, gradient.columns);
                this.v = new Matrix(this.m);
            }

            this.m.scale(this.beta1).add(new Matrix(gradient).scale(1 - this.beta1));
            this.v.scale(this.beta2).add(new Matrix(gradient).apply(val => val * val).scale(1 - this.beta2));

            const exp = Math.floor(this.i / this.batchSize) + 1;
            const mHat = new Matrix(this.m).scale(1 / (1 - Math.pow(this.beta1, exp))),
                vHat = new Matrix(this.v).scale(1 / (1 - Math.pow(this.beta2, exp)));

            gradient = mHat.scale(vHat.apply(Math.sqrt).add(this.epsilon).apply(val => 1 / val));

            callback(input, gradient);
        });
    }

}