import {
    type MinimalLanguageModel,
    predictNextTokenProbabilities,
} from '../09-minimal-trainable-language-model/index.js'
import { getGenerationContext, type TextGenerationTokenizer } from '../10-text-generation/index.js'

export type SamplingStrategy = 'greedy' | 'temperature' | 'topK'

export type SamplingSelectionOptions = {
    readonly strategy: SamplingStrategy
    readonly temperature?: number
    readonly topK?: number
    readonly random?: () => number
}

export type SamplingGenerationOptions = SamplingSelectionOptions & {
    readonly maxNewTokens: number
    readonly seed?: number
}

export type SamplingGenerationStep = {
    readonly step: number
    readonly strategy: SamplingStrategy
    readonly contextTokenIds: readonly number[]
    readonly selectedTokenId: number
    readonly selectedTokenProbability: number
    readonly tokenIdsAfterPrediction: readonly number[]
}

export type SamplingTokenGenerationResult = {
    readonly initialTokenIds: readonly number[]
    readonly generatedTokenIds: readonly number[]
    readonly tokenIds: readonly number[]
    readonly steps: readonly SamplingGenerationStep[]
}

export type SamplingTextGenerationResult = SamplingTokenGenerationResult & {
    readonly prompt: string
    readonly generatedText: string
    readonly text: string
}

const defaultTemperature = 1

/**
 * Ajuste une distribution avec une température.
 *
 * Intuition:
 * - température basse: les grosses probabilités deviennent encore plus dominantes;
 * - température haute: les petites probabilités ont davantage de chances d'être tirées.
 */
export function applyTemperature(
    probabilities: readonly number[],
    temperature: number,
): readonly number[] {
    validateProbabilityDistribution(probabilities, 'probabilities')
    validatePositiveNumber(temperature, 'temperature')

    const adjustedProbabilities = probabilities.map((probability) =>
        probability === 0 ? 0 : probability ** (1 / temperature),
    )

    return normalizeProbabilities(adjustedProbabilities)
}

/**
 * Garde seulement les k tokens les plus probables, puis renormalise.
 */
export function filterTopK(probabilities: readonly number[], topK: number): readonly number[] {
    validateProbabilityDistribution(probabilities, 'probabilities')
    validatePositiveInteger(topK, 'topK')

    if (topK > probabilities.length) {
        throw new Error(
            `topK doit être inférieur ou égal au nombre de probabilités. topK reçu: ${String(
                topK,
            )}, taille reçue: ${String(probabilities.length)}.`,
        )
    }

    const keptTokenIds = new Set(
        probabilities
            .map((probability, tokenId) => ({ probability, tokenId }))
            .sort((left, right) => right.probability - left.probability)
            .slice(0, topK)
            .map(({ tokenId }) => tokenId),
    )
    const filteredProbabilities = probabilities.map((probability, tokenId) =>
        keptTokenIds.has(tokenId) ? probability : 0,
    )

    return normalizeProbabilities(filteredProbabilities)
}

export function sampleFromProbabilities(
    probabilities: readonly number[],
    random: () => number = Math.random,
): number {
    validateProbabilityDistribution(probabilities, 'probabilities')

    const draw = random()

    if (!Number.isFinite(draw) || draw < 0 || draw >= 1) {
        throw new Error(
            `random doit retourner un nombre dans [0, 1). Valeur reçue: ${String(draw)}.`,
        )
    }

    let cumulativeProbability = 0

    for (const [tokenId, probability] of probabilities.entries()) {
        cumulativeProbability += probability

        if (draw < cumulativeProbability) {
            return tokenId
        }
    }

    return probabilities.length - 1
}

export function selectNextToken(
    probabilities: readonly number[],
    options: SamplingSelectionOptions,
): number {
    validateProbabilityDistribution(probabilities, 'probabilities')

    if (options.strategy === 'greedy') {
        return findMostLikelyToken(probabilities)
    }

    const temperature = options.temperature ?? defaultTemperature
    const adjustedProbabilities = applyTemperature(probabilities, temperature)

    if (options.strategy === 'temperature') {
        return sampleFromProbabilities(adjustedProbabilities, options.random)
    }

    if (options.topK === undefined) {
        throw new Error('topK est obligatoire pour la stratégie topK.')
    }

    const topKProbabilities = filterTopK(adjustedProbabilities, options.topK)

    return sampleFromProbabilities(topKProbabilities, options.random)
}

export function generateTokenIdsWithSampling(
    model: MinimalLanguageModel,
    initialTokenIds: readonly number[],
    options: SamplingGenerationOptions,
): SamplingTokenGenerationResult {
    validatePositiveInteger(options.maxNewTokens, 'maxNewTokens')

    const random = options.random ?? createDeterministicRandom(options.seed ?? 1234)
    const tokenIds = [...initialTokenIds]
    const generatedTokenIds: number[] = []
    const steps: SamplingGenerationStep[] = []

    for (let step = 1; step <= options.maxNewTokens; step++) {
        const contextTokenIds = getGenerationContext(tokenIds, model.contextLength)
        const probabilities = predictNextTokenProbabilities(model, contextTokenIds)
        const selectedTokenId = selectNextToken(
            probabilities,
            createSelectionOptions(options, random),
        )
        const selectedTokenProbability = readNumberAt(probabilities, selectedTokenId)

        tokenIds.push(selectedTokenId)
        generatedTokenIds.push(selectedTokenId)
        steps.push({
            contextTokenIds,
            selectedTokenId,
            selectedTokenProbability,
            step,
            strategy: options.strategy,
            tokenIdsAfterPrediction: [...tokenIds],
        })
    }

    return {
        generatedTokenIds,
        initialTokenIds: [...initialTokenIds],
        steps,
        tokenIds,
    }
}

function createSelectionOptions(
    options: SamplingGenerationOptions,
    random: () => number,
): SamplingSelectionOptions {
    return {
        random,
        strategy: options.strategy,
        ...(options.temperature === undefined ? {} : { temperature: options.temperature }),
        ...(options.topK === undefined ? {} : { topK: options.topK }),
    }
}

export function generateTextWithSampling(
    model: MinimalLanguageModel,
    tokenizer: TextGenerationTokenizer,
    prompt: string,
    options: SamplingGenerationOptions,
): SamplingTextGenerationResult {
    const initialTokenIds = tokenizer.encode(prompt)
    const tokenResult = generateTokenIdsWithSampling(model, initialTokenIds, options)
    const generatedText = tokenizer.decode([...tokenResult.generatedTokenIds])
    const text = tokenizer.decode([...tokenResult.tokenIds])

    return {
        ...tokenResult,
        generatedText,
        prompt,
        text,
    }
}

function createDeterministicRandom(seed: number): () => number {
    let state = normalizeSeed(seed)

    return () => {
        state = (state * 1664525 + 1013904223) % 4294967296

        return state / 4294967296
    }
}

function findMostLikelyToken(probabilities: readonly number[]): number {
    let bestTokenId = 0
    let bestProbability = readNumberAt(probabilities, 0)

    for (let tokenId = 1; tokenId < probabilities.length; tokenId++) {
        const probability = readNumberAt(probabilities, tokenId)

        if (probability > bestProbability) {
            bestTokenId = tokenId
            bestProbability = probability
        }
    }

    return bestTokenId
}

function normalizeProbabilities(probabilities: readonly number[]): readonly number[] {
    const sum = probabilities.reduce((total, probability) => total + probability, 0)

    if (!Number.isFinite(sum) || sum <= 0) {
        throw new Error('La somme des probabilités doit être strictement positive.')
    }

    return probabilities.map((probability) => probability / sum)
}

function normalizeSeed(seed: number): number {
    if (!Number.isFinite(seed)) {
        throw new Error(`seed doit être un nombre fini. Valeur reçue: ${String(seed)}.`)
    }

    return seed >>> 0
}

function readNumberAt(values: readonly number[], index: number): number {
    const value = values[index]

    if (value === undefined) {
        throw new Error(`Valeur introuvable à l'index ${String(index)}.`)
    }

    return value
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

function validateProbabilityDistribution(probabilities: readonly number[], name: string): void {
    if (probabilities.length === 0) {
        throw new Error(`${name} doit contenir au moins une probabilité.`)
    }

    for (const [index, probability] of probabilities.entries()) {
        if (!Number.isFinite(probability) || probability < 0) {
            throw new Error(
                `${name} doit contenir uniquement des nombres positifs ou nuls. Index ${String(
                    index,
                )}: ${String(probability)}.`,
            )
        }
    }

    normalizeProbabilities(probabilities)
}
