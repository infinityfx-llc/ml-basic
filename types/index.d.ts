export as namespace MLBasic;

import { FullyConnectedLayer } from './layers/fully-connected';
import { ConvolutionalLayer } from './layers/convolutional';

export { PreProcessor } from './pre-processor';

export { Neural } from './classifiers/neural';

export { Genetic } from './classifiers/genetic';

export { fromFile } from './utils/from-file';

export const layers = {
    fully_connected: FullyConnectedLayer,
    convolutional: ConvolutionalLayer
}

export const functions = {
    loss: object
}