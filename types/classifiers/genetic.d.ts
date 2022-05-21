import { Activation, HyperParameters } from '../global';
import { Classifier } from './classifier';

export class Genetic extends Classifier {

    static name = 'genetic';

    constructor({ shape, fitness_function, hyper_parameters, options }?: {
        shape?: number[] | {
            size: number;
            activation: Activation;
        }[];
        fitness_function?: (predict: (input: number[]) => number[]) => number;
        hyper_parameters?: HyperParameters;
        options?: Classifier.Options;
    });

    predict(input: number[], candidate?: number): Promise<number[]>;

    abstract fitness(): void;

    flush(): void;

    evaluate(): Promise<number[]>;

    evolve({ generations, hyper_parameters }?: {
        generations?: number;
        hyper_parameters?: HyperParameters;
    }): Promise<number>;

    tune(parameters?: object, { generations }?: {
        generations?: number
    }): Promise<object>;

    loadModel(model: object): Genetic;

    serialize(): string;

}