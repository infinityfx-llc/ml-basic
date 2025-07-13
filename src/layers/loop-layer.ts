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
    // outputs: Matrix[];
    // @ts-expect-error
    steps: {
        [key in T | 'output']: Matrix[];
    } = {};
    index = 0;

    constructor({
        input,
        output,
        activation = new Sigmoid()
    }: LoopParams) {
        super(input, [input[0], output], activation);

        this.state = new Matrix(input[0], 1);
        // this.outputs = new Array(Math.max(input[1], output) + 1);
    }

    abstract clear(): void;

    abstract forward(input: Matrix | undefined, output: boolean): Matrix | void;

    abstract backward(input: Matrix, output: Matrix, loss: Matrix): Matrix;

    cache(key: T | 'output', value: Matrix) {
        this.steps[key].push(new Matrix(value));
    }

    get(key: T | 'output', offset = 0) {
        return this.steps[key][this.index + offset];
    }

    propagate(input: Matrix) {
        input.reshape(...this.input);

        for (const key in this.steps) this.steps[key as T | 'output'] = []; // new

        const output = [],
            len = Math.max(this.input[1], this.output[1]);

        for (let i = 0; i < len; i++) {
            this.index = i; // new

            const stepInput = i < this.input[1] - 1 ?
                new Matrix(this.input[0], 1,
                    input.entries.slice(i * this.input[0], (i + 1) * this.input[0]) as any as number[]) :
                undefined;

            if (!i && stepInput) this.cache('output', stepInput); // new

            const stepOutput = this.forward(stepInput, i >= len - this.output[1]);
            // this.outputs[i + 1] = stepOutput ? stepOutput : new Matrix(this.state); // todo
            this.cache('output', stepOutput ? stepOutput : new Matrix(this.state));

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
            this.index = i; // new

            // const input = this.outputs[i],
            // output = this.outputs[i + 1].apply(this.activation.deactivate); // todo

            // loss = this.backward(input, output, loss);
            loss = this.backward(
                this.get('output'),
                this.get('output', 1).apply(this.activation.deactivate),
                loss);
            if (i < this.input[1]) inputLoss.push(...loss.entries);
        }

        return new Matrix(...this.input, inputLoss);
    }

}