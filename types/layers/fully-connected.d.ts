import { Matrix } from '../math/matrix';
import { Activator } from '../functions/activator';
import { Optimizer } from '../optimizers/activator';
import { HyperParameters } from '../global';
import { Layer } from './layer';

declare interface FullyConnectedLayer implements Layer {
    shape: {
        input: number;
        output: number;
    };
    weights: Matrix;
    bias: Matrix;
}

export class FullyConnectedLayer extends Layer {

    constructor({ input, output, activation, optimizer, hyper_parameters }?: {
        input?: number;
        output?: number;
        activation?: Activator;
        optimizer?: Optimizer;
        hyper_parameters?: HyperParameters;
    });

    shapeInput(data: number[] | Matrix): Matrix;

    propagate(input: Matrix): Matrix;

    backPropagate(input: Matrix, output: Matrix, loss: Matrix, hyper_parameters: HyperParameters): Matrix;

    cross(b: Layer): Layer;

    mutate({ mutation_probability, mutation_constant }?: {
        mutation_probability?: number;
        mutation_constant?: number;
    }): void;

    static deserialize(data: object): Layer;

}