import AveragePoolingLayer from "./average-pooling";
import ConvolutionalLayer from "./convolutional";
import FullyConnectedLayer from "./fully-connected";
import MaxPoolingLayer from "./max-pooling";
import RecurrentLayer from "./recurrent";

const Layers = {
    apol: AveragePoolingLayer,
    conv: ConvolutionalLayer,
    fcon: FullyConnectedLayer,
    mpol: MaxPoolingLayer,
    recu: RecurrentLayer
}

export default Layers;