import { Matrix } from '../math/matrix';
import { HyperParameters } from '../global';
import { Layer } from '../layer';
import { Loss } from '../global';
import { Log } from '../log';
import { Pool } from '../threading/pool';

declare interface Classifier {
    multithreading: boolean;
    pool: Pool | null;
}

export abstract class Classifier {

    constructor(multithreaded?: boolean);

    private parseShape;

    private createNetwork;

    static propagate(input: number[], network: Layer[]): Promise<Matrix[]>;

    static backPropagate(input: number[], target: number[], network: Layer[], { loss_function, hyper_parameters }?: {
        loss_function?: Loss;
        hyper_parameters?: HyperParameters;
    }): Promise<[number, Layer[]]>;

    static fit(data: {
        input: number[];
        target: number[];
    }[], network: Layer[], { loss_function, options }?: {
        loss_function?: Loss;
        options?: {
            max_epochs?: number;
            error_threshold?: number;
            iterative?: boolean;
            hyper_parameters?: HyperParameters;
        }
    }): Promise<[Log, Layer[]]>;

    abstract tune(iteration_function: (hyper_parameters: HyperParameters) => Promise<number>, parameters: object): Promise<object>;

    export(path: string): void;

    end(): void;

}

export namespace Classifier {

    interface Options {
        multithreaded?: boolean;
        binary?: boolean;
        labels?: string[]
    }

}