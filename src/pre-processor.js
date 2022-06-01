const Exception = require('./exceptions/exception');
const IllegalArgumentException = require('./exceptions/illegal-argument');
const utils = require('./utils');

// load csv and image files
// process text
// double array as batch inputs (recurrent)

module.exports = (() => {

    const fs = utils.isBrowser() ? {} : require('fs');

    return class PreProcessor {

        static inputKeys = ['data', 'input', 'in'];
        static targetKeys = ['target', 'output', 'out', 'label'];

        constructor() {
            this.labels = {
                __length: 0
            };
            this.shape = {
                input: {
                    max: 0
                },
                target: {
                    max: 0
                }
            };
            this.data = null;
        }

        async load(dataOrPath, target = null) {
            if (typeof dataOrPath === 'string') {
                const { extension } = (dataOrPath.match(/.*\.(?<extension>csv|json)$/i) || { groups: {} }).groups;
                if (extension === 'csv') return (await this.loadCSV(dataOrPath), this);
                if (extension === 'json') return (await this.loadJSON(dataOrPath), this);

                throw new IllegalArgumentException('Can only load files of type JSON or CSV');
            }

            return this.ingest(dataOrPath, target);
        }

        async dataFromFile(pathOrFile) {
            return new Promise((resolve, reject) => {
                if (utils.isBrowser()) {
                    if (!(pathOrFile instanceof Blob)) reject(new IllegalArgumentException('File must be an instance of File'));

                    let reader = new FileReader();
                    reader.onload = () => resolve(reader.result);
                    reader.onerror = () => reject(new IllegalArgumentException('File must be an instance of File'));
                    reader.readAsText(pathOrFile);
                } else {
                    fs.readFile(pathOrFile, (error, data) => {
                        if (error) return reject(new Exception('No such file'));
                        resolve(data);
                    });
                }
            });
        }

        async loadCSV(pathOrFile) {
            const data = await this.dataFromFile(pathOrFile);

            // Implement csv parsing

            return this;
        }

        async loadJSON(pathOrFile) {
            const data = await this.dataFromFile(pathOrFile);
            const json = JSON.parse(data);

            return this.ingest(json);
        }

        ingest(data, target = null) {
            if (!Array.isArray(data)) throw new IllegalArgumentException('Data must be an instance of Array');
            if (target && !Array.isArray(target)) throw new IllegalArgumentException('Target must be an instance of Array');
            if (target && target.length !== data.length) throw new IllegalArgumentException('Data length must match target length');

            this.data = data.map((val, i) => {
                if (target) {
                    const targetObject = Array.isArray(target[i]) || typeof target[i] !== 'object' ? { target: target[i] } : target[i];

                    val = Array.isArray(val) || typeof val !== 'object' ? { input: val } : val;
                    Object.assign(val, targetObject);
                } else {
                    val = Array.isArray(val) ? { input: val[0], target: val[1] } : val;
                }

                const input = val[this.findKey(val, PreProcessor.inputKeys)],
                    targetOrLabel = val[this.findKey(val, PreProcessor.targetKeys)];

                const isLabel = typeof targetOrLabel === 'string' && isNaN(parseFloat(targetOrLabel));
                if (isLabel && !(targetOrLabel in this.labels)) {
                    this.labels[targetOrLabel] = this.labels.__length;
                    this.labels.__length++;
                }

                return {
                    input: this.toArray(input),
                    target: isLabel ? targetOrLabel : this.toArray(targetOrLabel)
                };
            }).filter(val => !(typeof val.target !== 'string' && this.labels.__length));

            if (this.labels.__length > 0) {
                this.data = this.data.map(({ input, target }) => ({
                    input,
                    target: this.fromLabel(target)
                }));
            }

            return this;
        }

        fromLabel(label) {
            const bytes = Float64Array.BYTES_PER_ELEMENT * this.labels.__length,
                buffer = typeof SharedArrayBuffer !== 'undefined' ? new SharedArrayBuffer(bytes) : new ArrayBuffer(bytes),
                array = new Float64Array(buffer);
            array[this.labels[label]] = 1;

            return array;
        }

        toArray(value) {
            value = Array.isArray(value) ? value.flat() : [value];

            const bytes = value.length * Float64Array.BYTES_PER_ELEMENT,
                buffer = typeof SharedArrayBuffer !== 'undefined' ? new SharedArrayBuffer(bytes) : new ArrayBuffer(bytes),
                array = new Float64Array(buffer);
            for (let i = 0; i < array.length; i++) array[i] = Array.isArray(value) ? value[i] : value;

            return array;
        }

        findKey(object, keySet = PreProcessor.inputKeys) {
            for (const key of keySet) {
                if (key in object) return key;
            }

            return 0;
        }

        hash(object) {
            let hash = 0,
                str = JSON.stringify(object);

            for (let i = 0; i < str.length; i++) {
                hash = ((hash << 5) - hash) + str.charCodeAt(i);
                hash |= 0;
            }

            return hash;
        }

        trackSize({ input, target }) {
            input.length in this.shape.input ? this.shape.input[input.length]++ : this.shape.input[input.length] = 1;
            let len = this.shape.input[input.length];
            if (len > this.shape.input.max) this.shape.input.max = len, this.shape.input.key = input.length;

            target.length in this.shape.target ? this.shape.target[target.length]++ : this.shape.target[target.length] = 1;
            len = this.shape.target[target.length];
            if (len > this.shape.target.max) this.shape.target.max = len, this.shape.target.key = target.length;
        }

        clean({ nullToZero = true, removeDuplicates = true, allowVariableData = true } = {}) {
            const map = {};

            this.data = this.data
                .filter(val => {
                    if (val.input[0] === undefined || val.target[0] === undefined) return false;

                    const hasNaN = val.input.reduce((bool, val) => {
                        return (nullToZero && val === null ? bool : isNaN(parseFloat(val))) || bool;
                    }, false) ||
                        val.target.reduce((bool, val) => {
                            return (nullToZero && val === null ? bool : isNaN(parseFloat(val))) || bool;
                        }, false);
                    if (hasNaN) return false;

                    if (!removeDuplicates) {
                        if (!allowVariableData) this.trackSize(val);

                        return true;
                    }
                    const hash = this.hash(val);
                    if (hash in map) return false;
                    map[hash] = true;

                    if (!allowVariableData) this.trackSize(val);

                    return true;
                });

            if (nullToZero) this.data = this.data.map(val => {
                if (!nullToZero) return val;

                return {
                    input: val.input.map(val => (val === null ? 0 : val)),
                    target: val.target.map(val => (val === null ? 0 : val))
                };
            });

            if (!allowVariableData) this.data = this.data.filter(val => {
                return val.input.length === this.shape.input.key &&
                    val.target.length === this.shape.target.key
            });

            return this;
        }

        normalize(min = 0, max = 1) {
            let __max = this.max('input'),
                a_0 = (max - min) / (__max - this.min('input')),
                b_0 = max - a_0 * __max;

            __max = this.max('target');
            let a_1 = (max - min) / (__max - this.min('target')),
                b_1 = max - a_1 * __max;

            this.data = this.data.map(point => {
                return {
                    input: point.input.map(v => a_0 * v + b_0),
                    target: point.target.map(v => a_1 * v + b_1)
                }
            });

            return this;
        }

        min(key = 'input') {
            return this.data.reduce((min, val) => {
                const __min = utils.min(val[key]);
                return min >= __min ? __min : min;
            }, Number.MAX_VALUE);
        }

        max(key = 'input') {
            return this.data.reduce((max, val) => {
                const __max = utils.max(val[key]);
                return max <= __max ? __max : max;
            }, -Number.MAX_VALUE);
        }

        split() {
            //implement validation split
        }

        out() {
            return this.data;
        }

        getLabels() {
            const labels = new Array(this.labels.__length);
            Object.entries(this.labels).forEach(([label, i]) => label !== '__length' ? labels[i] = label : null);

            return labels;
        }

    }

})();