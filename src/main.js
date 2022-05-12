const { isBrowser } = require('./utils');

(function(root, factory) {
    if (typeof define === 'function' && define.amd) {
        define([], factory);
    } else
    if (typeof module === 'object' && module.exports) {
        module.exports = factory();
    } else {
        root.returnExports = factory();
    }
})(typeof self !== 'undefined' ? self : this, function() {
    return isBrowser() ? require('./es6') : require('./cjs');
});