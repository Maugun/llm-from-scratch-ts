export const projectName = 'typescript-llm'

export const moduleRoadmap = [
    '01-tokenizer-simple',
    '02-dataset-loader',
    '03-bigram-model',
    '04-embeddings',
    '05-self-attention',
    '06-transformer-block',
    '07-training-loop',
    '08-text-generation',
    '09-sampling-strategies',
    '10-vram-optimizations',
    '11-simple-tool-calling',
    '12-mcp-connectors-optional',
] as const

export { createCharacterTokenizer } from './modules/01-tokenizer-simple/index.js'
export type { CharacterTokenizer } from './modules/01-tokenizer-simple/index.js'

export {
    createTokenDataset,
    loadTextFile,
    loadTokenDatasetFromFile,
} from './modules/02-dataset-loader/index.js'
export type {
    TextTokenizer,
    TokenDataset,
    TokenDatasetOptions,
} from './modules/02-dataset-loader/index.js'
