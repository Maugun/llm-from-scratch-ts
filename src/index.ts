export const projectName = 'typescript-llm'

export const moduleRoadmap = [
    '01-tokenizer-simple',
    '02-dataset-loader',
    '03-bigram-model',
    '04-embeddings',
    '05-self-attention',
    '06-transformer-block',
    '07-positional-encoding',
    '08-training-loop-cpu',
    '09-minimal-trainable-language-model',
    '10-text-generation',
    '11-sampling-strategies',
    '12-tensorflowjs-autograd',
    '13-memory-performance-vram',
    '14-mini-end-to-end-pipeline',
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

export { createBigramModel } from './modules/03-bigram-model/index.js'
export type { BigramModel } from './modules/03-bigram-model/index.js'

export { cosineSimilarity, createEmbeddingTable } from './modules/04-embeddings/index.js'
export type { EmbeddingTable, EmbeddingTableOptions } from './modules/04-embeddings/index.js'

export {
    createSelfAttention,
    dotProduct,
    multiplyMatrixVector,
    softmax,
} from './modules/05-self-attention/index.js'
export type {
    AttentionApplication,
    SelfAttention,
    SelfAttentionOptions,
} from './modules/05-self-attention/index.js'

export {
    addVectors,
    applyFeedForward,
    createTransformerBlock,
    layerNormalize,
} from './modules/06-transformer-block/index.js'
export type {
    FeedForwardWeights,
    LayerNormalizeOptions,
    TransformerBlock,
    TransformerBlockApplication,
    TransformerBlockOptions,
} from './modules/06-transformer-block/index.js'

export {
    addPositionalEmbeddings,
    createPositionEmbeddingTable,
    getPositionEmbedding,
} from './modules/07-positional-encoding/index.js'
export type {
    PositionEmbeddingTable,
    PositionEmbeddingTableOptions,
} from './modules/07-positional-encoding/index.js'

export {
    createNextTokenExamples,
    createTrainableTokenBiasModel,
    crossEntropyLoss,
    perplexityFromLoss,
    softmax as softmaxTrainingLogits,
    trainNextTokenModel,
} from './modules/08-training-loop-cpu/index.js'
export type {
    NextTokenExample,
    NextTokenExampleOptions,
    TrainingEpochMetrics,
    TrainingHistory,
    TrainingOptions,
    TrainableTokenBiasModelOptions,
    TrainableNextTokenModel,
} from './modules/08-training-loop-cpu/index.js'

export {
    computeAverageLoss as computeMinimalLanguageModelAverageLoss,
    createMinimalLanguageModel,
    predictMostLikelyNextToken,
    predictNextTokenProbabilities as predictMinimalLanguageModelNextTokenProbabilities,
    trainMinimalLanguageModel,
} from './modules/09-minimal-trainable-language-model/index.js'
export type {
    MinimalLanguageModel,
    MinimalLanguageModelEpochMetrics,
    MinimalLanguageModelOptions,
    MinimalLanguageModelTrainingHistory,
} from './modules/09-minimal-trainable-language-model/index.js'
