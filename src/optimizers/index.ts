import GradientDescent from "./gradient-descent";
import BatchGradientDescent from "./batch-gradient-descent";
import Adam from "./adam";
import RMSProp from "./rms-prop";

const Optimizers = {
    sgd: GradientDescent,
    bgd: BatchGradientDescent,
    adam: Adam,
    rmsp: RMSProp
};

export default Optimizers;