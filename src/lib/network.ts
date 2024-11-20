import Layer from "../layers/layer";
import Optimizer from "../optimizers/optimizer";
import { LossFunction } from "./functions";
import Matrix from "./matrix";

export default class Network {

    layers: Layer[];
    lossFunction: LossFunction;

    constructor(layers: Layer[], optimizer: Optimizer, lossFunction: LossFunction) {
        this.lossFunction = lossFunction;
        this.layers = layers;

        for (let i = 0; i < layers.length - 1; i++) {
            const output = this.layers[i].output;
            const input = this.layers[i + 1].input;

            if (output[0] * output[1] !== input[0] * input[1]) throw new Error(`Layer ${i + 2}'s input does match layer ${i + 1}'s output`);

            this.layers[i].optimizer = optimizer.clone();
        }
    }

    propagate(input: Matrix) {
        let outputs = [input];

        for (let i = 0; i < this.layers.length; i++) outputs.push(this.layers[i].propagate(outputs[i]));

        return outputs;
    }

    backPropagate(input: Matrix, target: Matrix) {
        const outputs = this.propagate(input),
            output = outputs[outputs.length - 1],
            error = this.lossFunction.mean(output, target);

        let loss = this.lossFunction.derivative(output, target);

        for (let i = this.layers.length - 1; i >= 0; i--) {
            loss = this.layers[i].backPropagate(outputs[i], outputs[i + 1], loss);
        }

        return error;
    }

}