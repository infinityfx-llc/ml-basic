const isBrowser = typeof self !== 'undefined' && typeof self.navigator !== 'undefined' && typeof self.location !== 'undefined';

const __require = (url) => {
    if (!__require.stack) __require.stack = [self.location.href];
    const base = __require.stack[__require.stack.length - 1].replace(/[^\\\/]*?$/g, '');

    if (!/\.js$/i.test(url)) url += '.js';
    if (/^\.\.?(\/|\\)/.test(url)) {
        url = url.replace(/^\.(?:\/|\\)/, '');

        const count = (url.split(/\.\.(?:\/|\\)/) || [null]).length - 1,
            regx = new RegExp(`(?:(?:\\/|\\\\)?[^\\/\\\\]*?(?:\\/|\\\\)){${count}}$`, 'g');
        url = (count ? base.replace(regx, '/') : base) + url.replace(/\.\.(?:\/|\\)/g, '');
    } else {
        url = self.location.origin + '/' + url.replace(/^(\/|\\)/, '');
    }

    if (!__require.cache) __require.cache = {};
    let exports = __require.cache[url];

    if (!exports) {
        try {
            exports = {};
            const request = new XMLHttpRequest();
            request.open('GET', url, false);
            request.send();

            if (request.status !== 200) throw new Error(request.statusText);

            const module = { id: url, uri: url, exports };
            __require.stack.push(url);
            new Function('require', 'exports', 'module', request.responseText)(__require, exports, module);
            __require.cache[url] = exports = module.exports;
            __require.stack.pop();
        } catch (err) {
            throw new Error(`Error loading module ${url}: ${err}`);
        }
    }

    return exports;
};

const Task = isBrowser ? __require('./task') : require('./task');
const { parentPort } = isBrowser ? { parentPort: self } : require('worker_threads');

const onmessage = async (e) => {
    const { name, shared } = 'data' in e ? e.data : e;
    const result = await Task.run(name, shared);

    parentPort.postMessage(result);
};

isBrowser ? parentPort.onmessage = onmessage : parentPort.on('message', onmessage);