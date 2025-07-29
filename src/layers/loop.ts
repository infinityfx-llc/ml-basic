import { Activator, Sigmoid } from "../lib/functions";
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
        activation = new Sigmoid()
    }: LoopParams) {
        super(input, [input[0], output], activation);

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

        const output = [],
            len = Math.max(this.input[1], this.output[1]);

        for (let i = 0; i < len; i++) {
            this.index = i;

            const stepInput =
                new Matrix(this.input[0], 1,
                    i < this.input[1] - 1 ?
                        input.entries.slice(i * this.input[0], (i + 1) * this.input[0]) :
                        undefined);

            const stepOutput = this.forward(stepInput, i >= len - this.output[1]);
            if (stepOutput) output.push(...stepOutput.entries);
        }

        this.clear();

        return new Matrix(...this.output, output);
    }

    backPropagate(_1: Matrix, _2: Matrix, loss: Matrix) {
        const len = Math.max(this.input[1], this.output[1]),
            inputLoss = [];

        loss = new Matrix(this.output[0], 1,
            loss.entries.slice((this.output[1] - 1) * this.output[0]) as any as number[]);

        for (let i = len - 1; i >= 0; i--) {
            this.index = i;

            loss = this.backward(loss);
            if (i < this.input[1]) inputLoss.push(...loss.entries);
        }

        return new Matrix(...this.input, inputLoss);
    }

}