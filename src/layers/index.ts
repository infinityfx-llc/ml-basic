import { LayerArguments } from "../lib/network";
import AveragePoolingLayer from "./average-pooling";
import ConvolutionalLayer from "./convolutional";
import FullyConnectedLayer from "./fully-connected";
import MaxPoolingLayer from "./max-pooling";
import RecurrentLayer from "./recurrent";

const Layers = {
    avgp: (args: LayerArguments<typeof AveragePoolingLayer>) => ({
        layer: AveragePoolingLayer,
        args
    }),
    conv: (args: LayerArguments<typeof ConvolutionalLayer>) => ({
        layer: ConvolutionalLayer,
        args
    }),
    fcon: (args: LayerArguments<typeof FullyConnectedLayer>) => ({
        layer: FullyConnectedLayer,
        args
    }),
    maxp: (args: LayerArguments<typeof MaxPoolingLayer>) => ({
        layer: MaxPoolingLayer,
        args
    }),
    recu: (args: LayerArguments<typeof RecurrentLayer>) => ({
        layer: RecurrentLayer,
        args
    })
};

export default Layers;