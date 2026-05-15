export type SelfAttentionOptions = {
    /**
     * Dimension des vecteurs reçus en entrée.
     *
     * Elle doit correspondre à l'embeddingDimension du module 4.
     */
    readonly embeddingDimension: number

    /**
     * Dimension interne des queries, keys et values.
     *
     * Par défaut, on garde la même dimension que les embeddings pour limiter les surprises.
     */
    readonly attentionDimension?: number

    /**
     * Graine optionnelle pour créer les mêmes matrices Q/K/V à chaque exécution.
     */
    readonly seed?: number
}

export type AttentionApplication = {
    /**
     * Poids d'attention par position.
     *
     * attentionWeights[i][j] indique combien la position i regarde la position j.
     */
    readonly attentionWeights: readonly (readonly number[])[]

    /**
     * Scores avant softmax.
     *
     * Les positions futures sont marquées par -Infinity pour représenter le masque causal.
     */
    readonly attentionScores: readonly (readonly number[])[]

    /**
     * Vecteurs contextualisés produits par l'attention.
     */
    readonly outputVectors: readonly (readonly number[])[]
}

export type SelfAttention = {
    readonly embeddingDimension: number
    readonly attentionDimension: number
    readonly queryWeights: readonly (readonly number[])[]
    readonly keyWeights: readonly (readonly number[])[]
    readonly valueWeights: readonly (readonly number[])[]
    readonly applyCausalSelfAttention: (
        inputVectors: readonly (readonly number[])[],
    ) => AttentionApplication
}

const defaultSeed = 123
const initializationScale = 0.1

export function createSelfAttention(options: SelfAttentionOptions): SelfAttention {
    validatePositiveInteger(options.embeddingDimension, 'embeddingDimension')

    const attentionDimension = options.attentionDimension ?? options.embeddingDimension
    validatePositiveInteger(attentionDimension, 'attentionDimension')

    // Dans un vrai Transformer, ces matrices commencent souvent avec de petites valeurs
    // aléatoires, puis l'entraînement les ajuste par gradient. Ici nous n'avons pas encore
    // de training loop: elles restent donc aléatoires mais déterministes, seulement pour
    // rendre le mécanisme Q/K/V observable et reproductible.
    const random = createDeterministicRandom(options.seed ?? defaultSeed)
    const queryWeights = createProjectionMatrix(
        options.embeddingDimension,
        attentionDimension,
        random,
    )
    const keyWeights = createProjectionMatrix(
        options.embeddingDimension,
        attentionDimension,
        random,
    )
    const valueWeights = createProjectionMatrix(
        options.embeddingDimension,
        attentionDimension,
        random,
    )

    return {
        applyCausalSelfAttention: (inputVectors) =>
            applyCausalSelfAttention(inputVectors, {
                attentionDimension,
                embeddingDimension: options.embeddingDimension,
                keyWeights,
                queryWeights,
                valueWeights,
            }),
        attentionDimension,
        embeddingDimension: options.embeddingDimension,
        keyWeights,
        queryWeights,
        valueWeights,
    }
}

/**
 * Produit scalaire entre deux vecteurs.
 *
 * Dans l'attention, on compare une query avec une key via un produit scalaire:
 * plus le résultat est grand, plus la query "correspond" à cette key.
 */
export function dotProduct(vectorA: readonly number[], vectorB: readonly number[]): number {
    validateSameVectorDimension(vectorA, vectorB)

    return vectorA.reduce((sum, value, index) => sum + value * readNumberAt(vectorB, index), 0)
}

/**
 * Multiplie une matrice par un vecteur.
 *
 * Ici on s'en sert pour projeter un embedding vers Q, K ou V:
 * embedding -> query, embedding -> key, embedding -> value.
 */
export function multiplyMatrixVector(
    matrix: readonly (readonly number[])[],
    vector: readonly number[],
): number[] {
    return matrix.map((row) => dotProduct(row, vector))
}

/**
 * Transforme des scores arbitraires en probabilités positives qui somment à 1.
 *
 * Le softmax est central dans l'attention: après avoir calculé les scores de compatibilité,
 * il les transforme en poids de mélange. Les scores plus grands reçoivent plus de poids.
 */
export function softmax(values: readonly number[]): number[] {
    if (values.length === 0) {
        throw new Error('softmax attend au moins une valeur.')
    }

    const finiteValues = values.filter(Number.isFinite)

    if (finiteValues.length === 0) {
        throw new Error('softmax attend au moins une valeur finie.')
    }

    // Soustraire le maximum ne change pas le résultat mathématique du softmax, mais évite
    // des exponentielles énormes. C'est une astuce numérique très courante.
    const maxValue = Math.max(...finiteValues)
    const exponentials = values.map((value) =>
        Number.isFinite(value) ? Math.exp(value - maxValue) : 0,
    )
    const sumExponentials = exponentials.reduce((sum, value) => sum + value, 0)

    return exponentials.map((value) => value / sumExponentials)
}

type ApplySelfAttentionOptions = {
    readonly attentionDimension: number
    readonly embeddingDimension: number
    readonly keyWeights: readonly (readonly number[])[]
    readonly queryWeights: readonly (readonly number[])[]
    readonly valueWeights: readonly (readonly number[])[]
}

function applyCausalSelfAttention(
    inputVectors: readonly (readonly number[])[],
    options: ApplySelfAttentionOptions,
): AttentionApplication {
    validateInputVectors(inputVectors, options.embeddingDimension)

    // Chaque embedding est projeté en trois rôles:
    // - Query: ce que cette position cherche.
    // - Key: ce que cette position annonce aux autres.
    // - Value: l'information qui sera effectivement mélangée dans la sortie.
    const queries = inputVectors.map((vector) => multiplyMatrixVector(options.queryWeights, vector))
    const keys = inputVectors.map((vector) => multiplyMatrixVector(options.keyWeights, vector))
    const values = inputVectors.map((vector) => multiplyMatrixVector(options.valueWeights, vector))
    const attentionScores: number[][] = []
    const attentionWeights: number[][] = []
    const outputVectors: number[][] = []
    const scalingFactor = Math.sqrt(options.attentionDimension)

    for (let queryIndex = 0; queryIndex < queries.length; queryIndex++) {
        // queryIndex représente la position pour laquelle on construit une nouvelle
        // représentation contextualisée. On ne calcule pas "une attention globale":
        // on produit une sortie séparée pour chaque position de la séquence.
        const query = readVectorAt(queries, queryIndex)
        const scoresForPosition: number[] = []

        for (let keyIndex = 0; keyIndex < keys.length; keyIndex++) {
            if (keyIndex > queryIndex) {
                // Masque causal: une position n'a pas le droit de regarder le futur.
                // On met -Infinity pour que softmax donne ensuite un poids 0 à ces positions.
                scoresForPosition.push(Number.NEGATIVE_INFINITY)
            } else {
                const key = readVectorAt(keys, keyIndex)
                // Score d'attention:
                // on compare ce que la position courante cherche (query)
                // avec ce que la position candidate annonce (key).
                //
                // La division par sqrt(attentionDimension) garde les scores dans une plage
                // plus stable quand la dimension augmente, ce qui évite un softmax trop brutal.
                const score = dotProduct(query, key) / scalingFactor

                scoresForPosition.push(score)
            }
        }

        // Le softmax transforme les scores en poids positifs qui somment à 1.
        // Ces poids disent "combien regarder" chaque position autorisée.
        const weightsForPosition = softmax(scoresForPosition)
        const outputVector = createZeroVector(options.attentionDimension)

        for (let valueIndex = 0; valueIndex < values.length; valueIndex++) {
            const weight = readNumberAt(weightsForPosition, valueIndex)
            const value = readVectorAt(values, valueIndex)

            // Somme pondérée des values:
            // output_i = somme_j attentionWeight(i, j) * value_j.
            // C'est ici que l'information des positions autorisées est mélangée.
            for (let dimensionIndex = 0; dimensionIndex < outputVector.length; dimensionIndex++) {
                const currentOutputValue = readNumberAt(outputVector, dimensionIndex)

                outputVector[dimensionIndex] =
                    currentOutputValue + weight * readNumberAt(value, dimensionIndex)
            }
        }

        attentionScores.push(scoresForPosition)
        attentionWeights.push(weightsForPosition)
        outputVectors.push(outputVector)
    }

    return {
        attentionScores,
        attentionWeights,
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

function createProjectionMatrix(
    inputDimension: number,
    outputDimension: number,
    random: () => number,
): number[][] {
    return Array.from({ length: outputDimension }, () =>
        Array.from({ length: inputDimension }, () => (random() * 2 - 1) * initializationScale),
    )
}

function createZeroVector(size: number): number[] {
    return Array.from({ length: size }, () => 0)
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

function validateInputVectors(
    inputVectors: readonly (readonly number[])[],
    embeddingDimension: number,
): void {
    if (inputVectors.length === 0) {
        throw new Error('applyCausalSelfAttention attend au moins un vecteur.')
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
