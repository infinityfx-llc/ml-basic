const IllegalArgumentException = require('./exceptions/illegal-argument');
const { isBrowser } = require('./utils');

module.exports = (() => {
    let { performance } = isBrowser() ? {
        performance: {
            now: self.performance.now.bind(self.performance),
            timeOrigin: self.performance.timeOrigin || self.performance.timing.navigationStart
        }
    } : require('perf_hooks');
    if (typeof performance === 'undefined') performance = Date, performance.timeOrigin = Date.now();

    return class Log {

        constructor() {
            this.elapsed = performance.now();
            this.entries = [];
        }

        add(entry = {}) {
            this.entries.push({
                ...entry,
                timestampe: performance.timeOrigin + performance.now()
            });

            return this;
        }

        set(attributes = {}) {
            Object.entries(attributes).forEach(([key, val]) => {
                if (key === 'elapsed' || key === 'entries') throw new IllegalArgumentException(`${key} is a reserverd argument`);

                this[key] = val;
            });

            return this;
        }

        increment(attributes = {}) {
            Object.entries(attributes).forEach(([key, val]) => {
                if (key === 'elapsed' || key === 'entries') throw new IllegalArgumentException(`${key} is a reserverd argument`);

                key in this ? this[key] += val : this[key] = val;
            });

            return this;
        }

        end() {
            this.elapsed = performance.now() - this.elapsed;
            return this;
        }

    }

})();