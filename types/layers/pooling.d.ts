import { Matrix } from '../math/matrix';
import { Activator } from '../functions/activator';
import { HyperParameters } from '../global';
import { Layer } from './layer';

declare interface PoolingLayer implements Layer {
    shape: {
        input: number[];
        output: number[];
    };
    stride: number;
    size: number[];
}

export class PoolingLayer extends Layer {

    constructor({ input, size, stride, activation }?: {
        input?: number[];
        size?: number[];
        stride?: number;
        activation?: Activator;
    });

    clone(): PoolingLayer;

    shapeInput(data: number[] | Matrix): Matrix;

    propagate(input: Matrix): Matrix;

    backPropagate(input: Matrix, output: Matrix, loss: Matrix, hyper_parameters: HyperParameters): Matrix;

    cross(): PoolingLayer;

    mutate(): PoolingLayer;

    static deserialize(data: object): PoolingLayer;

}