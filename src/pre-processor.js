const IllegalArgumentException = require('./exceptions/illegal-argument');
const utils = require('./utils');

module.exports = class PreProcessor {

    static inputKeys = ['data', 'input', 'in'];
    static targetKeys = ['target', 'output', 'out'];

    constructor(data) {
        if (!Array.isArray(data)) throw new IllegalArgumentException('Data must be an instance of Array');

        this.data = data.map(val => {
            val = Array.isArray(val) ? { input: val[0], target: val[1] } : val;
            const input = val[this.findKey(val, PreProcessor.inputKeys)],
                target = val[this.findKey(val, PreProcessor.targetKeys)];

            return {
                input: Array.isArray(input) ? input : [input],
                target: Array.isArray(target) ? target : [target]
            };
        });
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

    clean({ nullToZero = true, removeDuplicates = true } = {}) {
        const map = {};

        this.data = this.data
            .filter(val => {
                if (val.input[0] === undefined || val.target[0] === undefined) return false;

                const hasNaN = val.input.reduce((bool, val) => {
                    return (nullToZero && val === null ? false : isNaN(parseFloat(val))) || bool;
                }, false) ||
                    val.target.reduce((bool, val) => {
                        return (nullToZero && val === null ? false : isNaN(parseFloat(val))) || bool;
                    }, false);
                if (hasNaN) return false;

                if (!removeDuplicates) return true;
                const hash = this.hash(val);
                if (hash in map) return false;
                map[hash] = true;

                return true;
            })
            .map(val => {
                if (!nullToZero) return val;

                return {
                    input: val.input.map(val => (val === null ? 0 : val)),
                    target: val.target.map(val => (val === null ? 0 : val))
                };
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

}