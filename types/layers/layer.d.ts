import { Activator } from '../functions/activator';
import { Optimizer } from '../optimizers/activator';
import { HyperParameters } from '../global';

declare interface Layer {
    activation: Activator;
    optimizer: Optimizer;
    hyper_parameters: HyperParameters;
}

export class Layer {

    static name: string;

    constructor(activation?: Activator, optimizer?: Optimizer, hyper_parameters?: HyperParameters);

    clone(): Layer;

    equals(layer: Layer): boolean;

    flush(): void;

    private shapeData;

    static cross(a: Layer, b: Layer): Layer;

    serialize(): object;

    static deserialize(data: object): Layer;

}