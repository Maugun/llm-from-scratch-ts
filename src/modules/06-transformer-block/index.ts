import {
    type AttentionApplication,
    createSelfAttention,
    multiplyMatrixVector,
} from '../05-self-attention/index.js'

export type TransformerBlockOptions = {
    /**
     * Dimension des vecteurs manipulés par le bloc.
     *
     * C'est la taille "publique" du bloc: l'entrée et la sortie gardent cette dimension pour
     * pouvoir empiler plusieurs blocs plus tard.
     */
    readonly embeddingDimension: number

    /**
     * Dimension interne de l'attention.
     *
     * Elle peut être différente de embeddingDimension, mais la sortie d'attention est ensuite
     * reprojetée vers embeddingDimension pour rendre la connexion résiduelle possible.
     */
    readonly attentionDimension?: number

    /**
     * Dimension cachée du feed-forward.
     *
     * Dans beaucoup de Transformers, cette dimension est plus grande que celle des embeddings:
     * cela donne au petit réseau local plus d'espace pour transformer chaque position.
     */
    readonly feedForwardDimension?: number

    /**
     * Graine optionnelle pour obtenir les mêmes poids à chaque exécution.
     */
    readonly seed?: number
}

export type FeedForwardWeights = {
    /**
     * Première projection: embeddingDimension -> feedForwardDimension.
     */
    readonly inputWeights: readonly (readonly number[])[]

    /**
     * Deuxième projection: feedForwardDimension -> embeddingDimension.
     */
    readonly outputWeights: readonly (readonly number[])[]
}

export type TransformerBlockApplication = {
    readonly normalizedAttentionInputVectors: readonly (readonly number[])[]
    readonly attentionOutputVectors: readonly (readonly number[])[]
    readonly attentionResidualVectors: readonly (readonly number[])[]
    readonly normalizedFeedForwardInputVectors: readonly (readonly number[])[]
    readonly feedForwardOutputVectors: readonly (readonly number[])[]
    readonly outputVectors: readonly (readonly number[])[]
    readonly attentionScores: AttentionApplication['attentionScores']
    readonly attentionWeights: AttentionApplication['attentionWeights']
}

export type TransformerBlock = {
    readonly embeddingDimension: number
    readonly attentionDimension: number
    readonly feedForwardDimension: number
    readonly attentionOutputWeights: readonly (readonly number[])[]
    readonly feedForwardWeights: FeedForwardWeights
    readonly applyTransformerBlock: (
        inputVectors: readonly (readonly number[])[],
    ) => TransformerBlockApplication
}

export type LayerNormalizeOptions = {
    /**
     * Petite valeur ajoutée à la variance pour éviter une division par zéro.
     */
    readonly epsilon?: number
}

const defaultSeed = 789
const defaultLayerNormEpsilon = 1e-5
const initializationScale = 0.1

/**
 * Crée un bloc Transformer CPU pédagogique.
 *
 * Un bloc Transformer est une unité réutilisable: il reçoit une séquence de vecteurs et
 * renvoie une séquence de vecteurs de même dimension. Cette stabilité de shape permet
 * d'empiler plusieurs blocs plus tard.
 */
export function createTransformerBlock(options: TransformerBlockOptions): TransformerBlock {
    validatePositiveInteger(options.embeddingDimension, 'embeddingDimension')

    const attentionDimension = options.attentionDimension ?? options.embeddingDimension
    const feedForwardDimension = options.feedForwardDimension ?? options.embeddingDimension * 4

    validatePositiveInteger(attentionDimension, 'attentionDimension')
    validatePositiveInteger(feedForwardDimension, 'feedForwardDimension')

    const random = createDeterministicRandom(options.seed ?? defaultSeed)
    const attention = createSelfAttention({
        attentionDimension,
        embeddingDimension: options.embeddingDimension,
        seed: options.seed ?? defaultSeed,
    })
    const attentionOutputWeights = createProjectionMatrix(
        attentionDimension,
        options.embeddingDimension,
        random,
    )
    const feedForwardWeights = createFeedForwardWeights(
        options.embeddingDimension,
        feedForwardDimension,
        random,
    )

    return {
        applyTransformerBlock: (inputVectors) =>
            applyTransformerBlock(inputVectors, {
                attention,
                attentionOutputWeights,
                embeddingDimension: options.embeddingDimension,
                feedForwardWeights,
            }),
        attentionDimension,
        attentionOutputWeights,
        embeddingDimension: options.embeddingDimension,
        feedForwardDimension,
        feedForwardWeights,
    }
}

/**
 * Additionne deux vecteurs dimension par dimension.
 *
 * C'est la base de la connexion résiduelle:
 * nouvelleVersion = ancienneVersion + correction.
 */
export function addVectors(vectorA: readonly number[], vectorB: readonly number[]): number[] {
    validateSameVectorDimension(vectorA, vectorB)

    return vectorA.map((value, index) => value + readNumberAt(vectorB, index))
}

/**
 * Normalise un vecteur autour de 0 avec une échelle stable.
 *
 * Pour des développeurs, on peut le voir comme une petite fonction de préparation de données:
 * avant d'appeler une brique sensible aux amplitudes, on remet les valeurs dans une plage plus
 * régulière. Ici la normalisation est locale: elle regarde uniquement les valeurs du vecteur.
 */
export function layerNormalize(
    vector: readonly number[],
    options: LayerNormalizeOptions = {},
): number[] {
    if (vector.length === 0) {
        throw new Error('layerNormalize attend au moins une valeur.')
    }

    const epsilon = options.epsilon ?? defaultLayerNormEpsilon

    if (!Number.isFinite(epsilon) || epsilon <= 0) {
        throw new Error(
            `epsilon doit être un nombre fini strictement positif. Valeur reçue: ${String(epsilon)}.`,
        )
    }

    // Moyenne: centre de gravité simple des valeurs.
    // Exemple: [2, 4, 6] -> moyenne 4.
    const mean = vector.reduce((sum, value) => sum + value, 0) / vector.length

    // Variance: moyenne des écarts au carré.
    // Elle mesure si les valeurs sont serrées ou très dispersées autour de la moyenne.
    const variance =
        vector.reduce((sum, value) => {
            const distanceFromMean = value - mean

            return sum + distanceFromMean * distanceFromMean
        }, 0) / vector.length

    // Écart-type: racine carrée de la variance.
    // On divise par cette valeur pour ramener l'échelle du vecteur vers quelque chose de stable.
    const standardDeviation = Math.sqrt(variance + epsilon)

    return vector.map((value) => (value - mean) / standardDeviation)
}

/**
 * Applique le feed-forward position par position.
 *
 * Contrairement à l'attention, cette fonction ne regarde pas les autres tokens. Elle transforme
 * seulement le vecteur courant avec une petite fonction non linéaire:
 * projection -> ReLU -> projection.
 */
export function applyFeedForward(vector: readonly number[], weights: FeedForwardWeights): number[] {
    validateFeedForwardWeights(weights, vector.length)

    const hiddenVector = multiplyMatrixVector(weights.inputWeights, vector)
    const activatedHiddenVector = hiddenVector.map(relu)

    return multiplyMatrixVector(weights.outputWeights, activatedHiddenVector)
}

type ApplyTransformerBlockOptions = {
    readonly attention: ReturnType<typeof createSelfAttention>
    readonly attentionOutputWeights: readonly (readonly number[])[]
    readonly embeddingDimension: number
    readonly feedForwardWeights: FeedForwardWeights
}

function applyTransformerBlock(
    inputVectors: readonly (readonly number[])[],
    options: ApplyTransformerBlockOptions,
): TransformerBlockApplication {
    validateInputVectors(inputVectors, options.embeddingDimension)

    // Étape 1: pre-norm avant l'attention.
    // On stabilise chaque position avant de calculer Q/K/V.
    const normalizedAttentionInputVectors = inputVectors.map((vector) => layerNormalize(vector))
    const attentionResult = options.attention.applyCausalSelfAttention(
        normalizedAttentionInputVectors,
    )

    // Étape 2: projection de sortie d'attention.
    // L'attention peut travailler dans attentionDimension, mais le résiduel doit revenir à
    // embeddingDimension pour pouvoir faire ancienneVersion + correction.
    const attentionOutputVectors = attentionResult.outputVectors.map((vector) =>
        multiplyMatrixVector(options.attentionOutputWeights, vector),
    )

    // Étape 3: première connexion résiduelle.
    // On conserve l'entrée originale et on ajoute ce que l'attention propose comme correction.
    const attentionResidualVectors = inputVectors.map((vector, index) =>
        addVectors(vector, readVectorAt(attentionOutputVectors, index)),
    )

    // Étape 4: pre-norm avant feed-forward.
    // Le feed-forward transforme chaque position indépendamment, sans mélange entre tokens.
    const normalizedFeedForwardInputVectors = attentionResidualVectors.map((vector) =>
        layerNormalize(vector),
    )
    const feedForwardOutputVectors = normalizedFeedForwardInputVectors.map((vector) =>
        applyFeedForward(vector, options.feedForwardWeights),
    )

    // Étape 5: deuxième connexion résiduelle.
    // Même principe: on garde l'état après attention, puis on ajoute la correction locale.
    const outputVectors = attentionResidualVectors.map((vector, index) =>
        addVectors(vector, readVectorAt(feedForwardOutputVectors, index)),
    )

    return {
        attentionOutputVectors,
        attentionResidualVectors,
        attentionScores: attentionResult.attentionScores,
        attentionWeights: attentionResult.attentionWeights,
        feedForwardOutputVectors,
        normalizedAttentionInputVectors,
        normalizedFeedForwardInputVectors,
        outputVectors,
    }
}

function createDeterministicRandom(seed: number): () => number {
    let state = normalizeSeed(seed)

    return () => {
        state = (state * 1664525 + 1013904223) % 4294967296

        return state / 4294967296
    }
}

function createFeedForwardWeights(
    embeddingDimension: number,
    feedForwardDimension: number,
    random: () => number,
): FeedForwardWeights {
    return {
        inputWeights: createProjectionMatrix(embeddingDimension, feedForwardDimension, random),
        outputWeights: createProjectionMatrix(feedForwardDimension, embeddingDimension, random),
    }
}

function createProjectionMatrix(
    inputDimension: number,
    outputDimension: number,
    random: () => number,
): number[][] {
    return Array.from({ length: outputDimension }, () =>
        Array.from({ length: inputDimension }, () => (random() * 2 - 1) * initializationScale),
    )
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

function readVectorAt(vectors: readonly (readonly number[])[], index: number): readonly number[] {
    const vector = vectors[index]

    if (vector === undefined) {
        throw new Error(`Vecteur introuvable à l'index ${String(index)}.`)
    }

    return vector
}

function relu(value: number): number {
    // ReLU signifie "Rectified Linear Unit".
    // Version développeur: si la valeur est positive, on la laisse passer; sinon on la coupe à 0.
    // Cette coupure introduit une non-linéarité: le feed-forward ne se limite pas à une seule
    // grosse multiplication de matrices équivalente.
    return Math.max(0, value)
}

function validateFeedForwardWeights(weights: FeedForwardWeights, embeddingDimension: number): void {
    validateMatrix(weights.inputWeights, 'inputWeights')
    validateMatrix(weights.outputWeights, 'outputWeights')

    const feedForwardDimension = weights.inputWeights.length
    const firstInputRow = readVectorAt(weights.inputWeights, 0)
    const firstOutputRow = readVectorAt(weights.outputWeights, 0)

    if (firstInputRow.length !== embeddingDimension) {
        throw new Error(
            `inputWeights doit accepter des vecteurs de ${String(
                embeddingDimension,
            )} dimensions. Dimension reçue: ${String(firstInputRow.length)}.`,
        )
    }

    if (firstOutputRow.length !== feedForwardDimension) {
        throw new Error(
            `outputWeights doit accepter des vecteurs de ${String(
                feedForwardDimension,
            )} dimensions. Dimension reçue: ${String(firstOutputRow.length)}.`,
        )
    }

    if (weights.outputWeights.length !== embeddingDimension) {
        throw new Error(
            `outputWeights doit produire des vecteurs de ${String(
                embeddingDimension,
            )} dimensions. Dimension reçue: ${String(weights.outputWeights.length)}.`,
        )
    }
}

function validateInputVectors(
    inputVectors: readonly (readonly number[])[],
    embeddingDimension: number,
): void {
    if (inputVectors.length === 0) {
        throw new Error('applyTransformerBlock attend au moins un vecteur.')
    }

    for (const [index, vector] of inputVectors.entries()) {
        if (vector.length !== embeddingDimension) {
            throw new Error(
                `Le vecteur d'entrée ${String(index)} doit avoir ${String(
                    embeddingDimension,
                )} dimensions. Dimension reçue: ${String(vector.length)}.`,
            )
        }
    }
}

function validateMatrix(matrix: readonly (readonly number[])[], name: string): void {
    if (matrix.length === 0) {
        throw new Error(`${name} doit contenir au moins une ligne.`)
    }

    const rowDimension = readVectorAt(matrix, 0).length

    if (rowDimension === 0) {
        throw new Error(`${name} doit contenir des lignes non vides.`)
    }

    for (const [index, row] of matrix.entries()) {
        if (row.length !== rowDimension) {
            throw new Error(
                `${name} doit contenir des lignes de même dimension. Ligne ${String(
                    index,
                )}: ${String(row.length)}, attendu: ${String(rowDimension)}.`,
            )
        }
    }
}

function validatePositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(
            `${name} doit être un entier strictement positif. Valeur reçue: ${String(value)}.`,
        )
    }
}

function validateSameVectorDimension(vectorA: readonly number[], vectorB: readonly number[]): void {
    if (vectorA.length !== vectorB.length) {
        throw new Error(
            `Les deux vecteurs doivent avoir la même dimension. Dimensions reçues: ${String(
                vectorA.length,
            )} et ${String(vectorB.length)}.`,
        )
    }
}
