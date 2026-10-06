import Matrix from "./matrix";
import { parseCSV, readFile, saveJsonFile } from "./utils";

const inputKeys = ['input', 'in'] as const;
const targetKeys = ['target', 'output', 'out', 'label'] as const;

type InputValue = number[] | number | undefined | null;
type TargetValue = InputValue | string;

type InputKeys = typeof inputKeys[number];
type TargetKeys = typeof targetKeys[number];

export type DataEntry = {
    [key in InputKeys | TargetKeys]?: key extends InputKeys ? InputValue : TargetValue;
};

type InputData = DataEntry[] | InputValue[];
type TargetData = DataEntry[] | TargetValue[];

export default class DataFrame {

    data: {
        input: Matrix;
        target: Matrix;
    }[];
    labels = new Map<string, number>();
    size: {
        input: number;
        target: number;
    } = {
            input: 0,
            target: 0
        };

    constructor(raw: InputData);
    constructor(input: InputData, target: TargetData);
    constructor(rawOrInput: InputData, target?: TargetData) {
        if (target && rawOrInput.length !== target.length) throw new Error('Input size does not match target size');

        this.data = rawOrInput.map((entry, i) => {
            let input = this.isValue(entry) ? this.toValue(entry) : this.getKey(entry, inputKeys),
                output: Matrix;

            if (target) {
                output = this.isValue(target[i]) ? this.toValue(target[i]) : this.getKey(target[i], targetKeys);
            } else if (!this.isValue(entry)) {
                output = this.getKey(entry, targetKeys);
            }

            this.size.input = input.entries.length;
            this.size.target = output!.entries.length;

            return {
                input,
                target: output!
            };
        });
    }

    private isValue(value: TargetValue | DataEntry) {
        return typeof value !== 'object' || Array.isArray(value) || value === null;
    }

    private toValue(value: TargetValue) {
        value = typeof value === 'number' ? [value] : value;

        if (!value) return new Matrix(0, 0);

        if (typeof value === 'string') {
            let index = this.labels.get(value);

            if (index === undefined) {
                index = this.labels.size;
                this.labels.set(value, index);
            }

            value = new Array(index).fill(0); // auto padding without .clean() call?
            value[index] = 1;
        }

        return new Matrix(value.length, 1, value);
    }

    private getKey(entry: DataEntry, keys: readonly (InputKeys | TargetKeys)[]) {
        for (const key of keys) {
            if (key in entry) return this.toValue(entry[key]);
        }

        return new Matrix(0, 0);
    }

    private getLength(lengths: Map<number, number>, maximum: boolean) {
        return maximum ?
            Math.max(...lengths.keys()) :
            Array.from(lengths.entries()).sort((a, b) => b[1] - a[1])[0][0];
    }

    getLabels() {
        return Array.from(this.labels.keys());
    }

    clean({
        missing = 'remove',
        duplicates = 'keep',
        uneven = 'remove'
    }: {
        missing?: 'remove' | 'zero';
        duplicates?: 'remove' | 'keep';
        uneven?: 'remove' | 'pad';
    } = {}) {
        const data = new Set<string>();
        const lengths = {
            input: new Map<number, number>(),
            target: new Map<number, number>()
        };

        this.data = this.data.filter(entry => {
            const inputLength = entry.input.entries.length;
            const targetLength = entry.target.entries.length;

            if (missing === 'remove' && !(inputLength && targetLength)) return false;
            if (duplicates === 'remove') {
                const key = entry.input.entries.toString()
                if (data.has(key)) return false;

                data.add(key);
            }

            if (inputLength) lengths.input.set(inputLength, (lengths.input.get(inputLength) || 0) + 1);
            if (targetLength) lengths.target.set(targetLength, (lengths.target.get(targetLength) || 0) + 1);

            return true;
        });

        const inputLength = this.getLength(lengths.input, uneven === 'pad');
        const targetLength = this.getLength(lengths.target, uneven === 'pad');

        this.data = this.data.filter(entry => {
            const entryInputLength = entry.input.entries.length,
                entryTargetLength = entry.target.entries.length;

            const unevenInput = entryInputLength !== inputLength,
                unevenTarget = entryTargetLength !== targetLength;

            if (uneven === 'remove' && (
                (unevenInput && entryInputLength !== 0) ||
                (unevenTarget && entryTargetLength !== 0)
            )) return false;

            if (unevenInput) entry.input = new Matrix(inputLength, 1, entry.input.entries);
            if (unevenTarget) entry.target = new Matrix(targetLength, 1, entry.target.entries);

            return true;
        });

        return this;
    }

    normalize({
        mode = 'target',
        min = 0,
        max = 1
    }: {
        /**
         * @default 'target'
         */
        mode?: 'target' | 'input' | 'both';
        /**
         * @default 0
         */
        min?: number;
        /**
         * @default 1
         */
        max?: number;
    } = {}) {
        let minInput = Number.MAX_VALUE,
            maxInput = -Number.MAX_VALUE,
            minTarget = Number.MAX_VALUE,
            maxTarget = -Number.MAX_VALUE;

        for (const { input, target } of this.data) {
            for (let i = 0; i < input.entries.length; i++) {
                const val = input.entries[i];

                minInput = Math.min(minInput, val);
                maxInput = Math.max(maxInput, val);
            }

            for (let i = 0; i < target.entries.length; i++) {
                const val = target.entries[i];

                minTarget = Math.min(minTarget, val);
                maxTarget = Math.max(maxTarget, val);
            }
        }

        const inputDiff = maxInput - minInput;
        const aInput = inputDiff === 0 ? 0 : (max - min) / inputDiff;
        const bInput = inputDiff === 0 ? min : max - aInput * maxInput;

        const targetDiff = maxTarget - minTarget;
        const aTarget = targetDiff === 0 ? 0 : (max - min) / targetDiff;
        const bTarget = targetDiff === 0 ? min : max - aTarget * maxTarget;

        for (const { input, target } of this.data) {
            if (mode !== 'target') {
                input.apply(val => inputDiff === 0 ? min : aInput * val + bInput);
            }

            if (mode !== 'input') {
                input.apply(val => targetDiff === 0 ? min : aTarget * val + bTarget);
            }
        }

        return this;
    }

    split(divide = .5) {
        const index = Math.floor(this.data.length * divide),
            training = new DataFrame([]),
            validation = new DataFrame([]);

        training.data = this.data.slice(0, index);
        training.labels = this.labels;
        training.size = this.size;
        validation.data = this.data.slice(index);
        validation.labels = this.labels;
        validation.size = this.size;

        return [
            training,
            validation
        ];
    }

    static async from(file: string | Blob) {
        const [_, ext] = (typeof file === 'string' ?
            file.match(/.*\.(csv|json)$/i) :
            file.type.match(/\/(csv|json)$/i)) || [];
        if (!ext) throw new Error('DataFrame only supports loading .csv or .json files');

        const data = await readFile(file);

        if (ext.toLowerCase() === 'json') {
            return new DataFrame(JSON.parse(data));
        } else {
            return new DataFrame(parseCSV(data));
        }
    }

    save(file: string) {
        const data = JSON.stringify(this.data.map(({ input, target }) => ({
            input: Array.from(input.entries),
            target: Array.from(target.entries)
        })));

        saveJsonFile(file, data);
    }

}