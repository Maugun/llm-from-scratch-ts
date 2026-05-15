import eslint from '@eslint/js'
import globals from 'globals'
import tseslint from 'typescript-eslint'
import eslintConfigPrettier from 'eslint-config-prettier'

export default tseslint.config(
    {
        ignores: ['dist/**', 'coverage/**', 'node_modules/**'],
    },
    eslint.configs.recommended,
    ...tseslint.configs.strictTypeChecked,
    eslintConfigPrettier,
    {
        languageOptions: {
            globals: globals.node,
            parserOptions: {
                projectService: true,
                tsconfigRootDir: import.meta.dirname,
            },
        },
        rules: {
            'no-console': ['error', { allow: ['info', 'warn', 'error'] }],
        },
    },
    {
        files: ['test/**/*.ts'],
        languageOptions: {
            globals: globals.vitest,
        },
    },
    {
        extends: [tseslint.configs.disableTypeChecked],
        files: ['**/*.js', '**/*.mjs'],
    },
)
