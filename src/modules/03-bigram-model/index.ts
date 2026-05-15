export type BigramModel = {
    /**
     * Taille du vocabulaire utilisée pour dimensionner les matrices.
     *
     * Avec un tokenizer caractère pédagogique, cette taille reste petite. Une matrice dense
     * est donc lisible et acceptable pour apprendre, même si elle serait trop coûteuse pour
     * de très grands vocabulaires.
     */
    readonly vocabularySize: number

    /**
     * Matrice des comptages.
     *
     * transitionCounts[current][next] contient le nombre de fois où `next` apparaît juste
     * après `current` dans la séquence d'apprentissage.
     */
    readonly transitionCounts: readonly (readonly number[])[]

    /**
     * Matrice des probabilités conditionnelles.
     *
     * transitionProbabilities[current][next] vaut P(next | current), c'est-à-dire:
     * count(current, next) / count(current, *).
     */
    readonly transitionProbabilities: readonly (readonly number[])[]

    /**
     * Nombre total de transitions observées.
     *
     * Pour une séquence de N tokens, il y a au maximum N - 1 transitions.
     */
    readonly totalTransitions: number

    readonly getNextTokenProbabilities: (currentTokenId: number) => readonly number[]
    readonly getTransitionCount: (currentTokenId: number, nextTokenId: number) => number
    readonly predictMostLikelyNextToken: (currentTokenId: number) => number | undefined
}

/**
 * Crée un modèle bigramme à partir d'une séquence d'ids.
 *
 * Un bigramme regarde uniquement le token courant pour prédire le token suivant. Il ne
 * comprend pas le sens du texte: il compte seulement les transitions observées.
 *
 * Formule centrale:
 * P(next | current) = count(current, next) / count(current, *)
 *
 * Mémoire / VRAM:
 * ce module est CPU-only et n'utilise aucun tenseur. La VRAM reste donc à 0. La RAM dépend
 * de vocabularySize x vocabularySize, car on stocke une matrice dense de comptages et une
 * matrice dense de probabilités.
 */
export function createBigramModel(
    tokenIds: readonly number[],
    vocabularySize: number,
): BigramModel {
    validateVocabularySize(vocabularySize)
    validateTokenIds(tokenIds, vocabularySize)

    const transitionCounts = createSquareMatrix(vocabularySize)
    let totalTransitions = 0

    for (let index = 0; index < tokenIds.length - 1; index++) {
        const currentTokenId = readTokenIdAt(tokenIds, index)
        const nextTokenId = readTokenIdAt(tokenIds, index + 1)
        const countsForCurrentToken = readMutableMatrixRow(transitionCounts, currentTokenId)
        const currentCount = readNumberAt(countsForCurrentToken, nextTokenId)

        countsForCurrentToken[nextTokenId] = currentCount + 1
        totalTransitions++
    }

    const transitionProbabilities = normalizeTransitionCounts(transitionCounts)

    return {
        getNextTokenProbabilities: (currentTokenId) =>
            getNextTokenProbabilities(currentTokenId, transitionProbabilities),
        getTransitionCount: (currentTokenId, nextTokenId) =>
            getTransitionCount(currentTokenId, nextTokenId, transitionCounts),
        predictMostLikelyNextToken: (currentTokenId) =>
            predictMostLikelyNextToken(currentTokenId, transitionProbabilities),
        totalTransitions,
        transitionCounts,
        transitionProbabilities,
        vocabularySize,
    }
}

function createSquareMatrix(size: number): number[][] {
    return Array.from({ length: size }, () => Array.from({ length: size }, () => 0))
}

function getNextTokenProbabilities(
    currentTokenId: number,
    transitionProbabilities: readonly (readonly number[])[],
): readonly number[] {
    validateTokenId(currentTokenId, transitionProbabilities.length)

    return readMatrixRow(transitionProbabilities, currentTokenId)
}

function getTransitionCount(
    currentTokenId: number,
    nextTokenId: number,
    transitionCounts: readonly (readonly number[])[],
): number {
    validateTokenId(currentTokenId, transitionCounts.length)
    validateTokenId(nextTokenId, transitionCounts.length)

    const countsForCurrentToken = readMatrixRow(transitionCounts, currentTokenId)

    return readNumberAt(countsForCurrentToken, nextTokenId)
}

function normalizeTransitionCounts(transitionCounts: readonly (readonly number[])[]): number[][] {
    return transitionCounts.map((countsForCurrentToken) => {
        const totalForCurrentToken = countsForCurrentToken.reduce((sum, count) => sum + count, 0)

        if (totalForCurrentToken === 0) {
            return countsForCurrentToken.map(() => 0)
        }

        return countsForCurrentToken.map((count) => count / totalForCurrentToken)
    })
}

function predictMostLikelyNextToken(
    currentTokenId: number,
    transitionProbabilities: readonly (readonly number[])[],
): number | undefined {
    const probabilities = getNextTokenProbabilities(currentTokenId, transitionProbabilities)
    let bestTokenId: number | undefined
    let bestProbability = 0

    for (const [tokenId, probability] of probabilities.entries()) {
        if (probability > bestProbability) {
            bestProbability = probability
            bestTokenId = tokenId
        }
    }

    return bestTokenId
}

function validateTokenId(tokenId: number, vocabularySize: number): void {
    if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= vocabularySize) {
        throw new Error(
            `tokenId doit être un entier entre 0 et ${String(
                vocabularySize - 1,
            )}. Valeur reçue: ${String(tokenId)}.`,
        )
    }
}

function readMatrixRow(
    matrix: readonly (readonly number[])[],
    rowIndex: number,
): readonly number[] {
    const row = matrix[rowIndex]

    if (row === undefined) {
        throw new Error(`Ligne de matrice introuvable pour l'id ${String(rowIndex)}.`)
    }

    return row
}

function readMutableMatrixRow(matrix: number[][], rowIndex: number): number[] {
    const row = matrix[rowIndex]

    if (row === undefined) {
        throw new Error(`Ligne de matrice introuvable pour l'id ${String(rowIndex)}.`)
    }

    return row
}

function readNumberAt(values: readonly number[], index: number): number {
    const value = values[index]

    if (value === undefined) {
        throw new Error(`Valeur introuvable à l'index ${String(index)}.`)
    }

    return value
}

function readTokenIdAt(tokenIds: readonly number[], index: number): number {
    const tokenId = tokenIds[index]

    if (tokenId === undefined) {
        throw new Error(`Token introuvable à l'index ${String(index)}.`)
    }

    return tokenId
}

function validateTokenIds(tokenIds: readonly number[], vocabularySize: number): void {
    for (const tokenId of tokenIds) {
        validateTokenId(tokenId, vocabularySize)
    }
}

function validateVocabularySize(vocabularySize: number): void {
    if (!Number.isInteger(vocabularySize) || vocabularySize <= 0) {
        throw new Error(
            `vocabularySize doit être un entier strictement positif. Valeur reçue: ${String(
                vocabularySize,
            )}.`,
        )
    }
}
