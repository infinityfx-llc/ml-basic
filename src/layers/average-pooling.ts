import Matrix from "../lib/matrix";
import PoolingLayer from "./pooling";

export default class AveragePoolingLayer extends PoolingLayer {

    name = 'AveragePooling';

    propagate(input: Matrix) {

        return Matrix.pool({
            matrix: input.reshape(...this.input),
            window: this.window,
            stride: this.stride,
            pooler: (avg, val) => avg + val / (this.window[0] * this.window[1])
        });
    }

}