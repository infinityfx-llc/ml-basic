import Matrix from "../lib/matrix";
import PoolingLayer from "./pooling";

export default class AveragePoolingLayer extends PoolingLayer {

    name = 'avgp';

    propagate(input: Matrix) {

        return Matrix.pool({
            matrix: input.reshape(...this.input),
            window: this.window,
            stride: this.stride,
            pooler: (avg, val) => avg + val / (this.window[0] * this.window[1])
        });
    }

    backPropagatePoolIndex(aggregate: number, _: number, index: number, indices: number[]): number {
        indices.push(index);

        return aggregate;
    }

}