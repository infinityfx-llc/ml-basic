import { Matrix } from '../math/matrix';
import { HyperParameters } from '../global';

export abstract class Classifier {
    
    constructor(multithreaded?: boolean);

    private parseShape;

    private createNetwork;

    static propagate(): Promise<Matrix[]>;

    static backPropagate(): Promise<number>;

    abstract tune(iteration_function: (hyper_parameters: HyperParameters) => Promise<number>, parameters: object): Promise<object>;

    export(path: string): void;

}

export namespace Classifier {

    interface Options {
        multithreaded?: boolean;
        binary?: boolean;
        labels?: string[]
    }

}