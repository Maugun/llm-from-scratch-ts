import {
    type MinimalLanguageModel,
    predictMostLikelyNextToken,
} from '../09-minimal-trainable-language-model/index.js'

export type TextGenerationTokenizer = {
    readonly encode: (text: string) => number[]
    readonly decode: (tokenIds: number[]) => string
}

export type TextGenerationOptions = {
    readonly maxNewTokens: number
}

export type GenerationStep = {
    readonly step: number
    readonly contextTokenIds: readonly number[]
    readonly predictedTokenId: number
    readonly tokenIdsAfterPrediction: readonly number[]
}

export type TokenGenerationResult = {
    readonly initialTokenIds: readonly number[]
    readonly generatedTokenIds: readonly number[]
    readonly tokenIds: readonly number[]
    readonly steps: readonly GenerationStep[]
}

export type TextGenerationResult = TokenGenerationResult & {
    readonly prompt: string
    readonly generatedText: string
    readonly text: string
}

/**
 * Retourne la fenêtre de contexte utilisée par le modèle.
 *
 * Un modèle entraîné avec contextLength = 4 ne sait prédire qu'à partir de 4 tokens. Pendant
 * la génération, on prend donc toujours les 4 derniers tokens disponibles.
 */
export function getGenerationContext(
    tokenIds: readonly number[],
    contextLength: number,
): readonly number[] {
    validatePositiveInteger(contextLength, 'contextLength')

    if (tokenIds.length < contextLength) {
        throw new Error(
            `La séquence doit contenir au moins ${String(
                contextLength,
            )} tokens pour construire le contexte. Nombre reçu: ${String(tokenIds.length)}.`,
        )
    }

    return tokenIds.slice(tokenIds.length - contextLength)
}

/**
 * Génère des token ids avec un greedy decoding.
 *
 * Greedy signifie: à chaque étape, on prend le token le plus probable. C'est simple et
 * déterministe, mais souvent répétitif. Les stratégies plus souples arriveront au module 11.
 */
export function generateTokenIds(
    model: MinimalLanguageModel,
    initialTokenIds: readonly number[],
    options: TextGenerationOptions,
): TokenGenerationResult {
    validatePositiveInteger(options.maxNewTokens, 'maxNewTokens')

    const tokenIds = [...initialTokenIds]
    const generatedTokenIds: number[] = []
    const steps: GenerationStep[] = []

    for (let step = 1; step <= options.maxNewTokens; step++) {
        const contextTokenIds = getGenerationContext(tokenIds, model.contextLength)
        const predictedTokenId = predictMostLikelyNextToken(model, contextTokenIds)

        tokenIds.push(predictedTokenId)
        generatedTokenIds.push(predictedTokenId)
        steps.push({
            contextTokenIds,
            predictedTokenId,
            step,
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

/**
 * Génère du texte complet à partir d'un prompt humain.
 *
 * Cette fonction est seulement un pont pratique:
 * texte -> ids -> génération en ids -> texte.
 */
export function generateText(
    model: MinimalLanguageModel,
    tokenizer: TextGenerationTokenizer,
    prompt: string,
    options: TextGenerationOptions,
): TextGenerationResult {
    const initialTokenIds = tokenizer.encode(prompt)
    const tokenResult = generateTokenIds(model, initialTokenIds, options)
    const generatedText = tokenizer.decode([...tokenResult.generatedTokenIds])
    const text = tokenizer.decode([...tokenResult.tokenIds])

    return {
        ...tokenResult,
        generatedText,
        prompt,
        text,
    }
}

function validatePositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(
            `${name} doit être un entier strictement positif. Valeur reçue: ${String(value)}.`,
        )
    }
}
