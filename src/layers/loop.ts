import { Activator, Initialization, Sigmoid } from "../lib/functions";
import Matrix from "../lib/matrix";
import Layer from "./layer";

export type LoopParams = {
    input: [number, number];
    output: number;
    // offset?
    /**
     * @default {@link Sigmoid}
     */
    activation?: Activator;
    /**
     * @default 'xavier'
     */
    initialization?: Initialization;
};

export default abstract class LoopLayer<T extends string = ''> extends Layer {

    state: Matrix;
    // @ts-expect-error
    cache: {
        [key in T]: Matrix[];
    } = {};
    index = 0;

    constructor({
        input,
        output,
        activation = new Sigmoid(),
        initialization = 'xavier'
    }: LoopParams) {
        super(input, [input[0], output], activation, initialization);

        this.state = new Matrix(input[0], 1);
        this.optimizer.configure({ batchSize: 1 });
    }

    abstract clear(): void;

    abstract forward(input: Matrix, output: boolean): Matrix | void;

    abstract backward(loss: Matrix): Matrix;

    store(key: T, value: Matrix) {
        if (!(key in this.cache)) this.cache[key] = [];

        this.cache[key].push(new Matrix(value));
    }

    get(key: T, offset = 0) {
        return this.cache[key][this.index + offset];
    }

    propagate(input: Matrix) {
        input.reshape(...this.input);

        for (const key in this.cache) this.cache[key] = [];

        const len = Math.max(this.input[1], this.output[1]);
        const result = new Matrix(this.output[0], this.output[1]);

        for (let i = 0; i < len; i++) {
            this.index = i;

            const stepInput = new Matrix(this.input[0], 1);
            if (i < this.input[1]) {
                for (let r = 0; r < this.input[0]; r++) {
                    stepInput.entries[r] = input.entries[r * this.input[1] + i];
                }
            }

            const isOutput = i >= len - this.output[1];
            const stepOutput = this.forward(stepInput, isOutput);

            if (isOutput && stepOutput) {
                const outIndex = i - (len - this.output[1]);
                for (let r = 0; r < this.output[0]; r++) {
                    result.entries[r * this.output[1] + outIndex] = stepOutput.entries[r];
                }
            }
        }

        this.clear();

        return result;
    }

    backPropagate(_1: Matrix, _2: Matrix, loss: Matrix) {
        loss.reshape(...this.output);

        const len = Math.max(this.input[1], this.output[1]);
        const inputGrad = new Matrix(this.input[0], this.input[1]);
        let recurrentLoss = new Matrix(this.output[0], 1);

        for (let i = len - 1; i >= 0; i--) {
            this.index = i;

            const stepLoss = new Matrix(this.output[0], 1);
            if (i >= len - this.output[1]) {
                const outIndex = i - (len - this.output[1]);
                for (let r = 0; r < this.output[0]; r++) {
                    stepLoss.entries[r] = loss.entries[r * this.output[1] + outIndex];
                }
            }
            stepLoss.add(recurrentLoss);

            recurrentLoss = this.backward(stepLoss);

            if (i < this.input[1]) {
                for (let r = 0; r < this.input[0]; r++) {
                    inputGrad.entries[r * this.input[1] + i] = recurrentLoss.entries[r];
                }
            }
        }

        return inputGrad;
    }

}