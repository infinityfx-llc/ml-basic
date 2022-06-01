import { PoolingLayer } from './pooling';

declare interface MaxPoolingLayer implements PoolingLayer {}

export class MaxPoolingLayer extends PoolingLayer {

    clone(): MaxPoolingLayer;

    private pool;

    private unpool;

    static deserialize(data: object): MaxPoolingLayer;

}