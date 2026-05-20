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
    '12-tensorflow-autograd',
    '13-tfjs-next-token-model',
    '14-trainable-mini-transformer',
    '15-model-sizing-memory-estimator',
    '16-tfjs-node-backend',
    '17-long-corpus-pipeline',
    '18-small-real-model-training',
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

export {
    generateText,
    generateTokenIds,
    getGenerationContext,
} from './modules/10-text-generation/index.js'
export type {
    GenerationStep,
    TextGenerationOptions,
    TextGenerationResult,
    TextGenerationTokenizer,
    TokenGenerationResult,
} from './modules/10-text-generation/index.js'

export {
    applyTemperature,
    filterTopK,
    generateTextWithSampling,
    generateTokenIdsWithSampling,
    sampleFromProbabilities,
    selectNextToken,
} from './modules/11-sampling-strategies/index.js'
export type {
    SamplingGenerationOptions,
    SamplingGenerationStep,
    SamplingSelectionOptions,
    SamplingStrategy,
    SamplingTextGenerationResult,
    SamplingTokenGenerationResult,
} from './modules/11-sampling-strategies/index.js'

export {
    createScalarRegressionModel,
    disposeScalarRegressionModel,
    meanSquaredError,
    predict,
    trainScalarRegressionModel,
} from './modules/12-tensorflow-autograd/index.js'
export type {
    ScalarRegressionExample,
    ScalarRegressionModel,
    ScalarRegressionOptions,
    TensorEpochMetrics,
    TensorTrainingHistory,
    TensorTrainingOptions,
} from './modules/12-tensorflow-autograd/index.js'

export {
    computeTfjsNextTokenAverageLoss,
    createTfjsNextTokenModel,
    disposeTfjsNextTokenModel,
    predictMostLikelyNextToken as predictMostLikelyTfjsNextToken,
    predictNextTokenLogits,
    predictNextTokenProbabilities as predictTfjsNextTokenProbabilities,
    trainTfjsNextTokenModel,
} from './modules/13-tfjs-next-token-model/index.js'
export type {
    TfjsNextTokenEpochMetrics,
    TfjsNextTokenModel,
    TfjsNextTokenModelOptions,
    TfjsNextTokenTrainingHistory,
    TfjsNextTokenTrainingOptions,
} from './modules/13-tfjs-next-token-model/index.js'

export {
    computeMiniTransformerAverageLoss,
    createTrainableMiniTransformer,
    disposeTrainableMiniTransformer,
    generateMiniTransformerText,
    generateMiniTransformerTokenIds,
    predictMiniTransformerLogits,
    predictMiniTransformerNextToken,
    predictMiniTransformerProbabilities,
    trainMiniTransformer,
} from './modules/14-trainable-mini-transformer/index.js'
export type {
    MiniTransformerGenerationOptions,
    MiniTransformerGenerationStep,
    MiniTransformerTextGenerationResult,
    MiniTransformerTextGenerationTokenizer,
    MiniTransformerTokenGenerationResult,
    MiniTransformerTrainingHistory,
    MiniTransformerTrainingOptions,
    TrainableMiniTransformer,
    TrainableMiniTransformerOptions,
} from './modules/14-trainable-mini-transformer/index.js'

export {
    compareModelSizes,
    estimateMiniTransformerSize,
    formatBytes,
    formatParameterCount,
} from './modules/15-model-sizing-memory-estimator/index.js'
export type {
    AttentionCostEstimate,
    MemoryEstimate,
    ModelSizeConfig,
    ModelSizeEstimate,
    NormalizedModelSizeConfig,
    ParameterGroupEstimate,
} from './modules/15-model-sizing-memory-estimator/index.js'
