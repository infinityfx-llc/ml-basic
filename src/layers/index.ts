import AveragePoolingLayer from "./average-pooling";
import ConvolutionalLayer from "./convolutional";
import FullyConnectedLayer from "./fully-connected";
import MaxPoolingLayer from "./max-pooling";
import RecurrentLayer from "./recurrent";

const Layers = {
    avgp: AveragePoolingLayer,
    conv: ConvolutionalLayer,
    fcon: FullyConnectedLayer,
    maxp: MaxPoolingLayer,
    recu: RecurrentLayer
}

export default Layers;