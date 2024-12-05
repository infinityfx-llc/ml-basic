import AveragePoolingLayer from "./average-pooling";
import ConvolutionalLayer, { ConvolutionalParams } from "./convolutional";
import FullyConnectedLayer, { FullyConnectedParams } from "./fully-connected";
import MaxPoolingLayer from "./max-pooling";
import { PoolingParams } from "./pooling";
import RecurrentLayer from "./recurrent";

type PartialParams<T extends { input: any; }> = Omit<T, 'input'> & Partial<Pick<T, 'input'>>;

const Layers = {
    avgp: (args: PartialParams<PoolingParams>) => ({
        layer: AveragePoolingLayer,
        args
    }),
    conv: (args: PartialParams<ConvolutionalParams>) => ({
        layer: ConvolutionalLayer,
        args
    }),
    fcon: (args: PartialParams<FullyConnectedParams>) => ({
        layer: FullyConnectedLayer,
        args
    }),
    maxp: (args: PartialParams<PoolingParams>) => ({
        layer: MaxPoolingLayer,
        args
    }),
    recu: (args: PartialParams<{ input: '' }>) => ({
        layer: RecurrentLayer,
        args
    })
};

export default Layers;