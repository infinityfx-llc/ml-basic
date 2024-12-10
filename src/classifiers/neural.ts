import { Layers } from "../layers";
import DataFrame from "../lib/data-frame";
import { LossFunction, SquaredLoss } from "../lib/functions";
import Matrix from "../lib/matrix";
import Network from "../lib/network";
import { range, shuffle } from "../lib/utils";
import BatchGradientDescent from "../optimizers/batch-gradient-descent";
import Optimizer from "../optimizers/optimizer";
import Classifier from "./classifier";

export default class Neural<O extends Optimizer> extends Classifier {

    name = 'Neural';
    epochs = 0;
    error = 1;
    network: Network;
    optimizer: O;

    constructor({
        layers,
        optimizer = new BatchGradientDescent() as any,
        lossFunction = new SquaredLoss()
    }: {
        layers: Layers[];
        optimizer?: O;
        lossFunction?: LossFunction;
    }) {
        super();

        this.network = new Network(layers, optimizer, lossFunction);
        this.optimizer = optimizer;
    }

    propagate(input: Matrix) {
        const outputs = this.network.propagate(input);

        return outputs[outputs.length - 1];
    }

    backPropagate(input: Matrix, target: Matrix) {
        return this.network.backPropagate(input, target);
    }

    predict({
        input,
        ranking = false,
        labels
    }: {
        input: Matrix | number[],
        ranking?: boolean;
        labels?: any[];
    }) {
        if (!(input instanceof Matrix)) input = new Matrix(input.length, 1, input);

        const output = this.propagate(input),
            array = Array.from(output.entries);

        if (!labels) return array;

        if (array.length !== labels.length) throw new Error('Label count is not equal to output size');

        const labeled = array.map((certainty, i) => ({
            certainty,
            label: labels[i]
        })).sort((a, b) => b.certainty - a.certainty);

        return ranking ? labeled : labeled[0];
    }

    fit({
        data,
        epochs,
        errorThreshold = 0,
        hyperParameters = {},
        logProgress = false
    }: {
        data: DataFrame;
        epochs: number;
        /**
         * @default 0
         */
        errorThreshold?: number;
        hyperParameters?: Omit<{
            [K in keyof O as O[K] extends Function ? never : K]?: O[K];
        }, 'name' | 't'>;
        logProgress?: boolean;
    }) {
        const batchSize = 'batchSize' in this.optimizer ? this.optimizer.batchSize as number : 1;
        epochs = Math.ceil(epochs / batchSize) * batchSize;

        this.network.configure(hyperParameters);

        if (logProgress) console.log(`\nFitting model (${data.data.length} samples / ${epochs} epochs):`);
        const start = performance.now();

        for (let i = 0; i < epochs; i++) {
            const order = shuffle(range(data.data.length));

            this.error = 0;
            for (let j = 0; j < order.length; j++) {
                const { input, target } = data.data[order[j]];

                const error = this.backPropagate(input, target);
                this.error += error / order.length;

                if (logProgress) {
                    const progress = (i * order.length + j + 1) / (epochs * order.length) * 100;
                    const loader = {
                        0: '-',
                        1: '\\',
                        2: '|',
                        3: '/'
                    }[Math.round(progress) % 4];

                    process.stdout.write(`\rCompletion: ${(progress).toFixed(1)}% / Error: ${(this.error * 100).toFixed(2)}% -> ${loader}`);
                }
            }

            this.epochs++;

            if (this.error <= errorThreshold) break;
        }

        if (logProgress) {
            const secs = (performance.now() - start) / 1000;
            process.stdout.write(`\rCompletion: 100.0% / Error: ${(this.error * 100).toFixed(2)}% -> ${secs.toFixed(1)}sec\n\r\n`);
        }

        return this.error;
    }

    validate({
        data
    }: {
        data: DataFrame;
    }) {
        const order = shuffle(range(data.data.length)),
            errors = {
                min: 1,
                avg: 0,
                max: 0
            };

        for (let j = 0; j < order.length; j++) {
            const { input, target } = data.data[order[j]];

            const error = this.backPropagate(input, target);
            errors.avg += error / order.length;
            errors.min = Math.min(errors.min, error);
            errors.max = Math.max(errors.max, error);
        }

        return errors;
    }

}