import { Matrix } from './math/matrix';
import { Activator } from './activators/activator';
import { Optimizer } from './optimizers/activator';
import { HyperParameters } from './global';

export class Layer {

    constructor(input?: number,
        output?: number,
        activation?: Activator,
        optimizer?: Optimizer,
        hyper_parameters?: HyperParameters);

    equals(layer: Layer): boolean;

    flush(): void;

    propagate(input: Matrix): Matrix;

    backPropagate(input: Matrix, output: Matrix, loss: Matrix, hyper_parameters: HyperParameters): Matrix;

    serialize(): object;

    static deserialize(data: object): Layer;

}