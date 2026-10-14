import { Layers } from "../layers";
import { LossFunction, SquaredLoss } from "../lib/functions";
import Matrix from "../lib/matrix";
import BatchGradientDescent from "../optimizers/batch-gradient-descent";
import Optimizer from "../optimizers/optimizer";
import Neural from "./neural";

export default class Autoencoder<O extends Optimizer = BatchGradientDescent> extends Neural<O> {

    name = 'Autoencoder';
    encoderSize: number;

    constructor({
        encoder,
        decoder,
        optimizer = new BatchGradientDescent() as any,
        lossFunction = new SquaredLoss()
    }: {
        encoder: Layers[];
        decoder: Layers[];
        optimizer?: O;
        lossFunction?: LossFunction;
    }) {
        const layers = [...encoder, ...decoder];
        super({ layers, optimizer, lossFunction });

        this.encoderSize = encoder.length;
    }

    encode({ input }: {
        input: Matrix | number[];
    }) {
        if (!(input instanceof Matrix)) input = new Matrix(input.length, 1, input);

        for (let i = 0; i < this.encoderSize; i++) {
            input = this.network.layers[i].propagate(input);
        }

        return input;
    }

    decode({ input }: {
        input: Matrix | number[];
    }) {
        if (!(input instanceof Matrix)) input = new Matrix(input.length, 1, input);

        for (let i = this.encoderSize; i < this.network.layers.length; i++) {
            input = this.network.layers[i].propagate(input);
        }

        return input;
    }

    reconstruct({ input }: {
        input: Matrix | number[];
    }) {
        if (!(input instanceof Matrix)) input = new Matrix(input.length, 1, input);

        return this.propagate(input);
    }

    /**
     * @deprecated Autoencoder is an unsupervised model without discrete class labels. Use reconstruct(), encode(), or decode() instead.
     */
    predict(): number[] {
        throw new Error('Autoencoder does not support predict(); use reconstruct(), encode(), or decode() instead');
    }

}
