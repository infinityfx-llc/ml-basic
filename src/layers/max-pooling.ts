import Matrix from "../lib/matrix";
import PoolingLayer from "./pooling";

export default class MaxPoolingLayer extends PoolingLayer {

    name = 'MaxPooling';

    propagate(input: Matrix) {

        return Matrix.pool({
            matrix: input.reshape(...this.input),
            window: this.window,
            stride: this.stride,
            initial: -Number.MAX_VALUE,
            pooler: (max, val) => Math.max(max, val)
        });
    }

}