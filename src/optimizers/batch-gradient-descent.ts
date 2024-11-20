import Matrix from "../lib/matrix";
import GradientDescent from "./gradient-descent";

export default class BatchGradientDescent extends GradientDescent {

    batchSize: number;
    aggregate?: Matrix;

    constructor({
        learningRate = 0.1,
        clipping = 0,
        batchSize = 8
    } = {}) {
        super({ learningRate, clipping });

        this.batchSize = batchSize;
    }

    step(gradient: Matrix) {
        this.aggregate ?
            this.aggregate.add(gradient) :
            this.aggregate = new Matrix(gradient);

        if (this.t % this.batchSize === 0) {
            gradient = super.step(this.aggregate.scale(1 / this.batchSize));
            this.aggregate = undefined;

            return gradient;
        }

        this.t++; // not super elegant
        return new Matrix(gradient).set(0);
    }

}