const { isBrowser } = require('../utils');

module.exports = (() => {

    const { Worker } = isBrowser() ? { Worker: self.Worker } : require('worker_threads');

    return class Pool {

        constructor(max_threads = 8) {
            this.file = (isBrowser() ? module.uri.replace(/[^\\\/]*?$/, '') : __dirname + '/') + 'thread.js';
            this.max_threads = max_threads;
            this.threads = 0;
            this.available = [];

            this.tasks = [];
        }

        onfreethread(callback) {
            this.tasks.push(callback);
        }

        addThread() {
            const worker = new Worker(this.file);

            const onmessage = e => {
                e = typeof e === 'object' && 'data' in e ? e.data : e;
                worker.resolve(e);

                this.available.push(worker);

                while (this.available.length && this.tasks.length) {
                    this.tasks.pop()();
                }
            };
            isBrowser() ? worker.onmessage = onmessage : worker.on('message', onmessage);

            const onerror = error => {
                worker.reject(error);
                this.threads--;
            };

            if (isBrowser()) {
                worker.onerror = onerror;
            } else {
                worker.on('error', onerror);
                worker.on('exit', code => code !== 0 ? onerror(new Error(`Exited with code ${code}`)) : null);
            }

            this.available.push(worker);
            this.threads++;
        }

        async queue(task) {
            return new Promise((resolve, reject) => {
                if (this.threads < this.max_threads) this.addThread();

                if (!this.available.length) {
                    this.onfreethread(function () {
                        const worker = this.available.pop();
                        worker.resolve = resolve;
                        worker.reject = reject;
                        worker.postMessage(task);
                    }.bind(this));

                    return;
                }

                const worker = this.available.pop();
                worker.resolve = resolve;
                worker.reject = reject;
                worker.postMessage(task);
            });
        }

        flush() {
            this.available.forEach(worker => worker.terminate());

            this.available = [];
        }

    };

})();