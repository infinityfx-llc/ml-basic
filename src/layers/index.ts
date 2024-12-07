import AveragePoolingLayer from "./average-pooling";
import ConvolutionalLayer, { ConvolutionalParams } from "./convolutional";
import FullyConnectedLayer, { FullyConnectedParams } from "./fully-connected";
import MaxPoolingLayer from "./max-pooling";
import { PoolingParams } from "./pooling";
import RecurrentLayer, { RecurrentParams } from "./recurrent";

type LayerParams<T extends { input: any; }> = Omit<T, 'input'> & Partial<Pick<T, 'input'>>;

const Layers = {
    avgp: (args: LayerParams<PoolingParams>) => ({
        Layer: AveragePoolingLayer,
        args
    }),
    conv: (args: LayerParams<ConvolutionalParams>) => ({
        Layer: ConvolutionalLayer,
        args
    }),
    fcon: (args: LayerParams<FullyConnectedParams>) => ({
        Layer: FullyConnectedLayer,
        args
    }),
    maxp: (args: LayerParams<PoolingParams>) => ({
        Layer: MaxPoolingLayer,
        args
    }),
    recu: (args: LayerParams<RecurrentParams>) => ({
        Layer: RecurrentLayer,
        args
    })
};

export type Layers = ReturnType<(typeof Layers)[keyof typeof Layers]>;

export default Layers;