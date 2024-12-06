import Layer from "../layers/layer";
import Optimizer, { HyperParameters } from "../optimizers/optimizer";
import { LossFunction } from "./functions";
import Matrix from "./matrix";

// todo (show actual object type):
export type LayerArguments<T extends abstract new (...args: any) => any> = Omit<ConstructorParameters<T>[0], 'input'> & Partial<Pick<ConstructorParameters<T>[0], 'input'>>;

export type LayerPreset<T extends typeof Layer = any> = {
    layer: T;
    args: LayerArguments<T>;
};

export default class Network {

    layers: Layer[];
    lossFunction: LossFunction;

    constructor(layers: LayerPreset[], optimizer: Optimizer, lossFunction: LossFunction) {
        this.lossFunction = lossFunction;
        this.layers = new Array(layers.length);

        for (let i = 0; i < layers.length; i++) {
            const output = i > 0 ? this.layers[i - 1].output : undefined;
            const { layer, args } = layers[i];

            if (!output && !('input' in args)) throw new Error(`Layer ${i + 1} has no input size defined`);

            this.layers[i] = new layer(Object.assign({ input: output }, args));
            this.layers[i].optimizer = optimizer.clone();
            const input = this.layers[i].input;

            if (output && output[0] * output[1] !== input[0] * input[1]) throw new Error(`Layer ${i + 1}'s input does match layer ${i}'s output`);
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

    configure(options: HyperParameters) {
        this.layers.forEach(layer => layer.optimizer.configure(options));
    }

}