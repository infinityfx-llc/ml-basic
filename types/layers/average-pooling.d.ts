import { PoolingLayer } from './pooling';

declare interface AveragePoolingLayer implements PoolingLayer {}

export class AveragePoolingLayer extends PoolingLayer {

    clone(): AveragePoolingLayer;

    private pool;

    private unpool;

    static deserialize(data: object): AveragePoolingLayer;

}