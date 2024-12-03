export function range(max: number): number[];
export function range(min: number, max: number): number[];
export function range(minOrMax: number, max?: number) {
    if (!max) max = minOrMax, minOrMax = 0;

    return new Array(max - minOrMax).fill(0).map((_, i) => minOrMax + i);
}

export function shuffle(array: any[]) {
    for (let i = array.length - 1; i > 0; i--) {
        const index = Math.floor(Math.random() * (i + 1));
        
        [array[i], array[index]] = [array[index], array[i]];
    }

    return array;
}

export function calculatePooledMatrix(rows: number, cols: number, kernel: number, stride: number, padding: number): [number, number] {
    return [
        Math.ceil((rows + padding * 2 - kernel) / stride + 1),
        Math.ceil((cols + padding * 2 - kernel) / stride + 1)
    ];
}