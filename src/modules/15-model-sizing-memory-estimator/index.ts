export type ModelSizeConfig = {
    readonly name?: string
    readonly vocabularySize: number
    readonly contextLength: number
    readonly embeddingDimension: number
    readonly feedForwardDimension: number
    readonly layerCount: number
    readonly batchSize?: number
    readonly bytesPerParameter?: number
}

export type ParameterGroupEstimate = {
    readonly tokenEmbeddings: number
    readonly positionEmbeddings: number
    readonly attentionPerLayer: number
    readonly feedForwardPerLayer: number
    readonly transformerBlockPerLayer: number
    readonly transformerBlocksTotal: number
    readonly outputProjection: number
    readonly total: number
}

export type MemoryEstimate = {
    readonly bytesPerParameter: number
    readonly parameterBytes: number
    readonly gradientBytes: number
    readonly adamStateBytes: number
    readonly trainingParameterBytes: number
    readonly attentionScoreBytesPerLayer: number
    readonly attentionScoreBytesTotal: number
    readonly roughActivationBytes: number
}

export type AttentionCostEstimate = {
    readonly scoresPerLayer: number
    readonly scoresTotal: number
}

export type ModelSizeEstimate = {
    readonly name: string | undefined
    readonly config: NormalizedModelSizeConfig
    readonly parameters: ParameterGroupEstimate
    readonly memory: MemoryEstimate
    readonly attention: AttentionCostEstimate
}

export type NormalizedModelSizeConfig = {
    readonly name: string | undefined
    readonly vocabularySize: number
    readonly contextLength: number
    readonly embeddingDimension: number
    readonly feedForwardDimension: number
    readonly layerCount: number
    readonly batchSize: number
    readonly bytesPerParameter: number
}

const defaultBytesPerParameter = 4
const defaultBatchSize = 1
const adamExtraBufferCount = 2
const activationBufferApproximationFactor = 6

export function estimateMiniTransformerSize(config: ModelSizeConfig): ModelSizeEstimate {
    const normalizedConfig = normalizeConfig(config)

    const tokenEmbeddings = normalizedConfig.vocabularySize * normalizedConfig.embeddingDimension
    const positionEmbeddings = normalizedConfig.contextLength * normalizedConfig.embeddingDimension

    // Une tête d'attention entraînable contient quatre projections carrées:
    // Q, K, V, puis la projection de sortie qui remet l'information mélangée
    // dans l'espace des embeddings.
    const attentionPerLayer = 4 * normalizedConfig.embeddingDimension ** 2

    // Le feed-forward est une petite fonction locale par position:
    // embedding -> dimension interne -> embedding.
    // On compte aussi les biais, car ce sont également des paramètres appris.
    const feedForwardPerLayer =
        normalizedConfig.embeddingDimension * normalizedConfig.feedForwardDimension +
        normalizedConfig.feedForwardDimension +
        normalizedConfig.feedForwardDimension * normalizedConfig.embeddingDimension +
        normalizedConfig.embeddingDimension

    const transformerBlockPerLayer = attentionPerLayer + feedForwardPerLayer
    const transformerBlocksTotal = transformerBlockPerLayer * normalizedConfig.layerCount

    const outputProjection =
        normalizedConfig.embeddingDimension * normalizedConfig.vocabularySize +
        normalizedConfig.vocabularySize

    const total = tokenEmbeddings + positionEmbeddings + transformerBlocksTotal + outputProjection

    const scoresPerLayer =
        normalizedConfig.batchSize * normalizedConfig.contextLength * normalizedConfig.contextLength
    const scoresTotal = scoresPerLayer * normalizedConfig.layerCount
    const parameterBytes = total * normalizedConfig.bytesPerParameter
    const gradientBytes = total * normalizedConfig.bytesPerParameter

    // Adam garde deux mémoires par paramètre: une moyenne des gradients et une
    // moyenne des gradients au carré. C'est très pratique pour l'apprentissage,
    // mais cela augmente fortement la mémoire d'entraînement.
    const adamStateBytes = total * normalizedConfig.bytesPerParameter * adamExtraBufferCount
    const attentionScoreBytesPerLayer = scoresPerLayer * normalizedConfig.bytesPerParameter
    const attentionScoreBytesTotal = scoresTotal * normalizedConfig.bytesPerParameter
    const roughActivationBytes =
        normalizedConfig.batchSize *
        normalizedConfig.contextLength *
        normalizedConfig.embeddingDimension *
        normalizedConfig.layerCount *
        normalizedConfig.bytesPerParameter *
        activationBufferApproximationFactor

    return {
        attention: {
            scoresPerLayer,
            scoresTotal,
        },
        config: normalizedConfig,
        memory: {
            adamStateBytes,
            attentionScoreBytesPerLayer,
            attentionScoreBytesTotal,
            bytesPerParameter: normalizedConfig.bytesPerParameter,
            gradientBytes,
            parameterBytes,
            roughActivationBytes,
            trainingParameterBytes: parameterBytes + gradientBytes + adamStateBytes,
        },
        name: normalizedConfig.name,
        parameters: {
            attentionPerLayer,
            feedForwardPerLayer,
            outputProjection,
            positionEmbeddings,
            tokenEmbeddings,
            total,
            transformerBlockPerLayer,
            transformerBlocksTotal,
        },
    }
}

export function compareModelSizes(
    configs: readonly ModelSizeConfig[],
): readonly ModelSizeEstimate[] {
    return configs.map((config) => estimateMiniTransformerSize(config))
}

export function formatBytes(bytes: number): string {
    validateNonNegativeFiniteNumber(bytes, 'bytes')

    const units = ['B', 'KB', 'MB', 'GB', 'TB'] as const
    let value = bytes
    let unitIndex = 0

    while (value >= 1024 && unitIndex < units.length - 1) {
        value /= 1024
        unitIndex++
    }

    if (unitIndex === 0) {
        return `${String(value)} B`
    }

    return `${value.toFixed(2)} ${readUnit(units, unitIndex)}`
}

export function formatParameterCount(parameterCount: number): string {
    validateNonNegativeFiniteNumber(parameterCount, 'parameterCount')

    if (parameterCount < 1_000) {
        return String(parameterCount)
    }

    if (parameterCount < 1_000_000) {
        return `${(parameterCount / 1_000).toFixed(2)}K`
    }

    if (parameterCount < 1_000_000_000) {
        return `${(parameterCount / 1_000_000).toFixed(2)}M`
    }

    return `${(parameterCount / 1_000_000_000).toFixed(2)}B`
}

function normalizeConfig(config: ModelSizeConfig): NormalizedModelSizeConfig {
    validatePositiveInteger(config.vocabularySize, 'vocabularySize')
    validatePositiveInteger(config.contextLength, 'contextLength')
    validatePositiveInteger(config.embeddingDimension, 'embeddingDimension')
    validatePositiveInteger(config.feedForwardDimension, 'feedForwardDimension')
    validatePositiveInteger(config.layerCount, 'layerCount')

    const batchSize = config.batchSize ?? defaultBatchSize
    const bytesPerParameter = config.bytesPerParameter ?? defaultBytesPerParameter

    validatePositiveInteger(batchSize, 'batchSize')
    validatePositiveNumber(bytesPerParameter, 'bytesPerParameter')

    return {
        batchSize,
        bytesPerParameter,
        contextLength: config.contextLength,
        embeddingDimension: config.embeddingDimension,
        feedForwardDimension: config.feedForwardDimension,
        layerCount: config.layerCount,
        name: config.name,
        vocabularySize: config.vocabularySize,
    }
}

function validatePositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(
            `${name} doit être un entier strictement positif. Valeur reçue: ${String(value)}.`,
        )
    }
}

function validatePositiveNumber(value: number, name: string): void {
    if (!Number.isFinite(value) || value <= 0) {
        throw new Error(
            `${name} doit être un nombre fini strictement positif. Valeur reçue: ${String(value)}.`,
        )
    }
}

function validateNonNegativeFiniteNumber(value: number, name: string): void {
    if (!Number.isFinite(value) || value < 0) {
        throw new Error(
            `${name} doit être un nombre fini positif ou nul. Valeur reçue: ${String(value)}.`,
        )
    }
}

function readUnit(units: readonly string[], index: number): string {
    const unit = units[index]

    if (unit === undefined) {
        throw new Error(`Unité de taille introuvable à l’index ${String(index)}.`)
    }

    return unit
}
