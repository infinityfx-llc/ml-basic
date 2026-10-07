import { DataEntry } from "./data-frame";
import { readFile as fsReadFile, writeFileSync } from 'fs';

export const browser = () => typeof window !== 'undefined' && typeof document !== 'undefined';

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

export function calculatePooledMatrix(rows: number, cols: number, kernel: number | [number, number], stride: number, padding: number): [number, number] {
    const [kRows, kCols] = Array.isArray(kernel) ? kernel : [kernel, kernel];

    return [
        Math.floor((rows + padding * 2 - kRows) / stride + 1),
        Math.floor((cols + padding * 2 - kCols) / stride + 1)
    ];
}

export async function readFile(file: string | Blob): Promise<string> {
    if (typeof file !== 'string') return file.text();

    if (browser()) {
        throw new Error('Unable to access file system from the browser');
    }

    return new Promise((resolve, reject) => {
        fsReadFile(file, 'utf-8', (error, data) => {
            if (error) reject(error);
            else resolve(data);
        });
    });
}

export function parseCSV(data: string) {
    const sanitize = (str: string) => str.trim().replace(/^["'](.*)["']$/, '$1');

    function parseValue(value: string) {
        const num = parseFloat(value);

        if (!isNaN(num)) return num;
        return value === 'null' ? null : value;
    }

    const [header, ...lines] = data
        .split(/\r?\n/g)
        .filter(line => !!line);
    const delimiter = header.includes(';') ? ';' : ',';
    const keys = header.split(delimiter).map(sanitize);

    return lines.map(line => {
        const values = line.split(delimiter);

        return values.reduce((entry, value, i) => {
            value = sanitize(value);
            const [_, array] = value.match(/^\[(.*)\]$/) || [];

            // @ts-expect-error
            entry[keys[i]] = array ? array.split(',').map(parseValue) : parseValue(value);

            return entry;
        }, {} as DataEntry);
    });
}

export function saveJsonFile(file: string, data: any) {
    if (!/\.json$/i.test(file)) file = file + '.json';
    const content = typeof data === 'string' ? data : JSON.stringify(data);

    if (browser()) {
        file = file.replace(/.*\//, '');

        const blob = new Blob([content], { type: 'application/json' }),
            a = document.createElement('a');
        a.href = URL.createObjectURL(blob);
        a.download = file;
        a.click();
        URL.revokeObjectURL(a.href);
    } else {
        writeFileSync(file, content, 'utf-8');
    }
}