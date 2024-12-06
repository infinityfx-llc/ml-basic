import { Activator } from "../lib/functions";
import Matrix from "../lib/matrix";
import GradientDescent from "../optimizers/gradient-descent";
import Optimizer from "../optimizers/optimizer";

export default abstract class Layer {

    type = 'Layer';
    abstract name: string;
    input: [number, number];
    output: [number, number];
    activation: Activator;
    optimizer: Optimizer = new GradientDescent();

    constructor(input: [number, number], output: [number, number], activation: Activator) {
        this.input = input;
        this.output = output;
        this.activation = activation;
    }

    abstract propagate(input: Matrix): Matrix;

    abstract backPropagate(input: Matrix, output: Matrix, loss: Matrix): Matrix;

}