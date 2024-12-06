import { DataEntry } from "./data-frame";
import { readFile as fsReadFile } from 'fs';

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

export async function readFile(file: string | Blob): Promise<string> {
    const browser = typeof self !== 'undefined' && typeof self.location !== 'undefined',
        blob = typeof file !== 'string';

    return new Promise((resolve, reject) => {
        if (blob) {

            const reader = new FileReader();
            reader.onload = () => resolve(reader.result as string);
            reader.onerror = reject;
            reader.readAsText(file, 'utf-8');
        }

        if (!browser && !blob) {
            fsReadFile(file, 'utf-8', (error, data) => {
                if (error) reject(error);

                resolve(data);
            });
        }

        if (browser && !blob) throw new Error('Unable to access file system from the browser');
    });
}

export function parseCSV(data: string) { // better errors for when this fails?
    const [header, ...lines] = data.split(/\r?\n/g);
    const keys = header.split(/;/g);

    function parseValue(value: string) {
        const num = parseFloat(value);

        if (!isNaN(num)) return num;
        return value === 'null' ? null : value;
    }

    return lines.map(line => {
        const values = line.split(/;/g);

        return values.reduce((entry, value, i) => {
            value = value.replace(/\"(.*)\"/, '$1');
            const [_, array] = value.match(/^\[(.*)\]$/) || [];

            // @ts-expect-error
            entry[keys[i]] = array ? array.split(',').map(parseValue) : parseValue(value);

            return entry;
        }, {} as DataEntry);
    });
}