import {
    crossEntropyLoss,
    perplexityFromLoss,
    softmax,
    type NextTokenExample,
    type TrainingOptions,
} from '../08-training-loop-cpu/index.js'

export type MinimalLanguageModelOptions = {
    readonly vocabularySize: number
    readonly contextLength: number
}

export type MinimalLanguageModel = {
    readonly vocabularySize: number
    readonly contextLength: number

    /**
     * Préférence globale pour chaque prochain token.
     *
     * C'est l'équivalent du modèle du module 8: même sans regarder le contexte, certains tokens
     * peuvent être plus fréquents que d'autres.
     */
    readonly outputBias: number[]

    /**
     * Influence du contexte.
     *
     * contextWeights[position][tokenDansLeContexte][prochainToken] répond à la question:
     * "si je vois ce token à cette position du contexte, est-ce que ce prochain token devient
     * plus ou moins probable ?"
     */
    readonly contextWeights: number[][][]
}

export type MinimalLanguageModelEpochMetrics = {
    readonly epoch: number
    readonly averageLoss: number
    readonly perplexity: number
}

export type MinimalLanguageModelTrainingHistory = {
    readonly initialLoss: number
    readonly finalLoss: number
    readonly epochs: readonly MinimalLanguageModelEpochMetrics[]
}

/**
 * Crée un modèle de langage minimal entraînable.
 *
 * Tous les poids commencent à 0. Avant entraînement, le modèle ne préfère donc aucun token:
 * les logits sont tous égaux et le softmax donne une distribution uniforme.
 */
export function createMinimalLanguageModel(
    options: MinimalLanguageModelOptions,
): MinimalLanguageModel {
    validatePositiveInteger(options.vocabularySize, 'vocabularySize')
    validatePositiveInteger(options.contextLength, 'contextLength')

    return {
        contextLength: options.contextLength,
        contextWeights: createZeroContextWeights(options.contextLength, options.vocabularySize),
        outputBias: Array.from({ length: options.vocabularySize }, () => 0),
        vocabularySize: options.vocabularySize,
    }
}

/**
 * Prédit P(nextToken | contexte).
 *
 * La différence importante avec le module 8 est ici: les logits ne viennent plus seulement d'un
 * biais global. Ils reçoivent aussi les contributions des tokens présents dans le contexte.
 */
export function predictNextTokenProbabilities(
    model: MinimalLanguageModel,
    inputTokenIds: readonly number[],
): readonly number[] {
    const logits = computeConditionalLogits(model, inputTokenIds)

    return softmax(logits)
}

export function predictMostLikelyNextToken(
    model: MinimalLanguageModel,
    inputTokenIds: readonly number[],
): number {
    const probabilities = predictNextTokenProbabilities(model, inputTokenIds)
    let bestTokenId = 0
    let bestProbability = readNumberAt(probabilities, 0)

    for (let tokenId = 1; tokenId < probabilities.length; tokenId++) {
        const probability = readNumberAt(probabilities, tokenId)

        if (probability > bestProbability) {
            bestProbability = probability
            bestTokenId = tokenId
        }
    }

    return bestTokenId
}

export function computeAverageLoss(
    model: MinimalLanguageModel,
    examples: readonly NextTokenExample[],
): number {
    validateExamples(model, examples)

    const lossSum = examples.reduce((sum, example) => {
        const probabilities = predictNextTokenProbabilities(model, example.inputTokenIds)

        return sum + crossEntropyLoss(probabilities, example.targetTokenId)
    }, 0)

    return lossSum / examples.length
}

/**
 * Entraîne le modèle avec une descente de gradient très explicite.
 *
 * Pour chaque exemple:
 * 1. on calcule les probabilités;
 * 2. on compare avec la bonne cible;
 * 3. on corrige le biais global;
 * 4. on corrige uniquement les poids liés aux tokens réellement vus dans le contexte.
 */
export function trainMinimalLanguageModel(
    model: MinimalLanguageModel,
    examples: readonly NextTokenExample[],
    options: TrainingOptions,
): MinimalLanguageModelTrainingHistory {
    validateTrainingOptions(options)
    validateExamples(model, examples)

    const initialLoss = computeAverageLoss(model, examples)
    const epochMetrics: MinimalLanguageModelEpochMetrics[] = []

    for (let epoch = 1; epoch <= options.epochs; epoch++) {
        for (const example of examples) {
            const probabilities = predictNextTokenProbabilities(model, example.inputTokenIds)

            for (let nextTokenId = 0; nextTokenId < model.vocabularySize; nextTokenId++) {
                const predictedProbability = readNumberAt(probabilities, nextTokenId)
                const expectedProbability = nextTokenId === example.targetTokenId ? 1 : 0
                const gradient = predictedProbability - expectedProbability

                model.outputBias[nextTokenId] =
                    readNumberAt(model.outputBias, nextTokenId) - options.learningRate * gradient

                for (const [position, contextTokenId] of example.inputTokenIds.entries()) {
                    const positionWeights = readContextPositionWeights(model, position)
                    const contextTokenWeights = readContextTokenWeights(
                        positionWeights,
                        contextTokenId,
                    )

                    // Les contributions du contexte sont moyennées dans les logits, donc le
                    // gradient de chaque poids de contexte est divisé par contextLength.
                    contextTokenWeights[nextTokenId] =
                        readNumberAt(contextTokenWeights, nextTokenId) -
                        (options.learningRate * gradient) / model.contextLength
                }
            }
        }

        const averageLoss = computeAverageLoss(model, examples)

        epochMetrics.push({
            averageLoss,
            epoch,
            perplexity: perplexityFromLoss(averageLoss),
        })
    }

    return {
        epochs: epochMetrics,
        finalLoss: computeAverageLoss(model, examples),
        initialLoss,
    }
}

function computeConditionalLogits(
    model: MinimalLanguageModel,
    inputTokenIds: readonly number[],
): number[] {
    validateContext(model, inputTokenIds)

    return model.outputBias.map((bias, nextTokenId) => {
        const contextContribution = inputTokenIds.reduce((sum, contextTokenId, position) => {
            const positionWeights = readContextPositionWeights(model, position)
            const contextTokenWeights = readContextTokenWeights(positionWeights, contextTokenId)

            return sum + readNumberAt(contextTokenWeights, nextTokenId)
        }, 0)

        return bias + contextContribution / model.contextLength
    })
}

function createZeroContextWeights(contextLength: number, vocabularySize: number): number[][][] {
    return Array.from({ length: contextLength }, () =>
        Array.from({ length: vocabularySize }, () =>
            Array.from({ length: vocabularySize }, () => 0),
        ),
    )
}

function readContextPositionWeights(model: MinimalLanguageModel, position: number): number[][] {
    const positionWeights = model.contextWeights[position]

    if (positionWeights === undefined) {
        throw new Error(`Poids introuvables pour la position ${String(position)}.`)
    }

    return positionWeights
}

function readContextTokenWeights(positionWeights: number[][], contextTokenId: number): number[] {
    const contextTokenWeights = positionWeights[contextTokenId]

    if (contextTokenWeights === undefined) {
        throw new Error(`Poids introuvables pour le token de contexte ${String(contextTokenId)}.`)
    }

    return contextTokenWeights
}

function readNumberAt(values: readonly number[], index: number): number {
    const value = values[index]

    if (value === undefined) {
        throw new Error(`Valeur introuvable à l'index ${String(index)}.`)
    }

    return value
}

function validateContext(model: MinimalLanguageModel, inputTokenIds: readonly number[]): void {
    validateModelShape(model)

    if (inputTokenIds.length !== model.contextLength) {
        throw new Error(
            `Le contexte doit contenir ${String(
                model.contextLength,
            )} tokens. Nombre reçu: ${String(inputTokenIds.length)}.`,
        )
    }

    for (const [position, tokenId] of inputTokenIds.entries()) {
        validateTokenId(tokenId, model.vocabularySize, `inputTokenIds[${String(position)}]`)
    }
}

function validateExamples(
    model: MinimalLanguageModel,
    examples: readonly NextTokenExample[],
): void {
    if (examples.length === 0) {
        throw new Error('Le modèle attend au moins un exemple.')
    }

    for (const example of examples) {
        validateContext(model, example.inputTokenIds)
        validateTokenId(example.targetTokenId, model.vocabularySize, 'targetTokenId')
    }
}

function validateModelShape(model: MinimalLanguageModel): void {
    validatePositiveInteger(model.vocabularySize, 'model.vocabularySize')
    validatePositiveInteger(model.contextLength, 'model.contextLength')

    if (model.outputBias.length !== model.vocabularySize) {
        throw new Error(
            `outputBias doit contenir ${String(model.vocabularySize)} valeurs. Nombre reçu: ${String(
                model.outputBias.length,
            )}.`,
        )
    }

    if (model.contextWeights.length !== model.contextLength) {
        throw new Error(
            `contextWeights doit contenir ${String(
                model.contextLength,
            )} positions. Nombre reçu: ${String(model.contextWeights.length)}.`,
        )
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

function validateTokenId(tokenId: number, vocabularySize: number, name: string): void {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabularySize) {
        throw new Error(
            `${name} doit être un entier entre 0 et ${String(
                vocabularySize - 1,
            )}. Valeur reçue: ${String(tokenId)}.`,
        )
    }
}

function validateTrainingOptions(options: TrainingOptions): void {
    validatePositiveInteger(options.epochs, 'epochs')
    validatePositiveNumber(options.learningRate, 'learningRate')
}
