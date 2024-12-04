import DataFrame from "../lib/data-frame";
import { LossFunction, SquaredLoss } from "../lib/functions";
import Matrix from "../lib/matrix";
import Network, { LayerPreset } from "../lib/network";
import { range, shuffle } from "../lib/utils";
import BatchGradientDescent from "../optimizers/batch-gradient-descent";
import Optimizer from "../optimizers/optimizer";
import Classifier from "./classifier";

export default class Neural<O extends Optimizer> extends Classifier {

    epochs = 0;
    error = 1;
    network: Network;
    optimizer: O;

    constructor({
        layers,
        optimizer = new BatchGradientDescent() as any,
        lossFunction = new SquaredLoss()
    }: {
        layers: LayerPreset<any>[];
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
        hyperParameters = {}
    }: {
        data: DataFrame;
        epochs: number;
        errorThreshold?: number;
        hyperParameters?: Omit<{
            [K in keyof O as O[K] extends Function ? never : K]?: O[K];
        }, 'name' | 't'>;
    }) {
        const batchSize = 'batchSize' in this.optimizer ? this.optimizer.batchSize as number : 1;
        epochs = Math.ceil(epochs / batchSize) * batchSize;

        this.network.configure(hyperParameters);

        for (let i = 0; i < epochs; i++) {
            const order = shuffle(range(data.data.length));

            this.error = 0;
            for (let j = 0; j < order.length; j++) {
                const { input, target } = data.data[order[j]];

                const error = this.backPropagate(input, target);
                this.error += error / order.length;
            }

            this.epochs++;
            if (this.error <= errorThreshold) break;
        }

        return this.error;
    }

}