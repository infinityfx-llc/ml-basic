export const range = (min = 0, max) => {
    if (!max) max = min, min = 0;
    return new Array(max - min).fill(0).map((_, i) => i);
};

export const shuffle = (array) => {
    for (let i = array.length - 1; i > 0; i--) {
        const index = Math.floor(Math.random() * (i + 1));
        [array[i], array[index]] = [array[index], array[i]];
    }

    return array;
};

export const pad = (array, size) => {
    return new Array(size).fill(0).map((_, i) => {
        return i < array.length ? array[i] : array[array.length - 1];
    });
};

export const combine = (a, b) => {
    b = pad(b, a.length);
    return a.map((val, idx) => ([val, b[idx]]));
};

export const argmin = (array) => {
    let min = Number.MAX_VALUE;

    return array.reduce((index, n, i) => n <= min ? (min = n, i) : index, 0);
};