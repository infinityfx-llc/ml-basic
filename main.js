const __require = (url) => {
    if (!__require.stack) __require.stack = [document.currentScript.src];
    const base = __require.stack[__require.stack.length - 1].replace(/[^\\\/]*?$/, '');

    if (!/\.js$/i.test(url)) url += '.js';
    if (/^\.\.?(\/|\\)/.test(url)) {
        url = /\/|\\/.test(url.charAt(1)) ? base + url.slice(2) : base + url;
    } else {
        url = window.location.origin + url;
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

(function(root, factory) {
    if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else
    if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.MLBasic = factory();
    }
})(typeof self !== 'undefined' ? self : this, function() {
    return typeof window !== 'undefined' && typeof window.document !== 'undefined' ? (window.require = __require, require('./src/es6')) : require('./src/cjs');
});