import GradientDescent, { GradientDescentParams } from "./gradient-descent";
import BatchGradientDescent, { BatchGradientDescentParams } from "./batch-gradient-descent";
import Adam, { AdamParams } from "./adam";
import RMSProp, { RMSPropParams } from "./rms-prop";

const Optimizers = {
    sgd: (args: GradientDescentParams) => new GradientDescent(args),
    bgd: (args: BatchGradientDescentParams) => new BatchGradientDescent(args),
    adam: (args: AdamParams) => new Adam(args),
    rmsp: (args: RMSPropParams) => new RMSProp(args)
};

export default Optimizers;