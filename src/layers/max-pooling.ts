import Matrix from "../lib/matrix";
import PoolingLayer from "./pooling";

export default class MaxPoolingLayer extends PoolingLayer {

    name = 'maxp';

    propagate(input: Matrix) {

        return Matrix.pool({
            matrix: input.reshape(...this.input),
            window: this.window,
            stride: this.stride,
            initial: -Number.MAX_VALUE,
            pooler: (max, val) => Math.max(max, val)
        });
    }

    backPropagatePoolIndex(max: number, value: number, index: number, indices: number[]): number {
        if (value > max) {
            indices[0] = index;
            return value;
        }

        return max;
    }

}