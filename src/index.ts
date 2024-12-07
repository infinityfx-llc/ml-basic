import { readFile } from "./lib/utils";
import * as Functions from "./lib/functions";
import Neural from "./classifiers/neural";
import Layers from "./layers";
import Optimizers from "./optimizers";
import DataFrame from "./lib/data-frame";
import Matrix from "./lib/matrix";

export * from './lib/functions';

export {
    Neural,
    Layers,
    Optimizers,
    DataFrame,
    Matrix
}

function deserialize(data: any): any {
    if (typeof data === 'object') {
        if (Array.isArray(data)) return data.map(deserialize);

        if (!('type' in data)) {
            for (const key in data) {
                let value;
                // @ts-expect-error
                if (key === 'lossFunction' || key === 'activation') value = new Functions[data[key]]();

                data[key] = value ? value : deserialize(data[key]);
            }

            return data;
        }

        if (data.type === 'Matrix') return new Matrix(data.rows, data.columns, data.entries); // conversion from normal array to float64 (may lose data)

        const { type, ...rest } = data;
        // @ts-expect-error
        if (type === 'Optimizer') return Object.assign(Optimizers[data.name](), deserialize(rest));
        if (type === 'Layer') {
            // @ts-expect-error
            const { Layer } = Layers[data.name]();
            return Object.assign(Object.create(Layer.prototype), deserialize(rest));
        }
    }

    return data;
}

export async function load(file: string | Blob): Promise<Neural<any>> {
    const data = await readFile(file).then(JSON.parse);

    switch (data.name) {
        case 'Neural':
            return Object.assign(Object.create(Neural.prototype), deserialize(data));
        default:
            throw new Error('Unable to load classifier');
    }
}