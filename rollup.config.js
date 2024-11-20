import terser from '@rollup/plugin-terser';
import del from 'rollup-plugin-delete';
import typescript from '@rollup/plugin-typescript';

const isProd = process.env.NODE_ENV === 'production';

export default {
    input: ['src/index.ts'],
    external: ['tslib'],
    output: {
        dir: 'dist',
        format: 'es',
        sourcemap: true
    },
    plugins: [
        isProd ?
            del({
                targets: 'dist/**'
            }) :
            undefined,
        typescript({
            tsconfig: './tsconfig.json'
        }),
        isProd ?
            terser({ compress: { directives: false } }) :
            undefined
    ]
}