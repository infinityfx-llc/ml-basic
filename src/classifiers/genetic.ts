import Matrix from "../lib/matrix";
import Classifier from "./classifier";

export default class Genetic extends Classifier {

    name = 'Genetic';

    constructor() {
        super();

        throw new Error('Not yet implemented');
    }

    propagate(input: Matrix): Matrix {
        return new Matrix(1, 1);
    }

    backPropagate(input: Matrix, target: Matrix): number {
        return 1;
    }

}