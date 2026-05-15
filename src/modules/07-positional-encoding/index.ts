export type PositionEmbeddingTableOptions = {
    /**
     * Longueur maximale de séquence supportée.
     *
     * Une table de positions a une ligne par position possible: position 0, position 1, etc.
     */
    readonly maxSequenceLength: number

    /**
     * Dimension de chaque embedding de position.
     *
     * Elle doit correspondre à la dimension des token embeddings pour pouvoir faire:
     * tokenEmbedding + positionEmbedding.
     */
    readonly embeddingDimension: number

    /**
     * Graine optionnelle pour obtenir la même table à chaque exécution.
     */
    readonly seed?: number
}

export type PositionEmbeddingTable = {
    readonly vectors: readonly (readonly number[])[]
    readonly maxSequenceLength: number
    readonly embeddingDimension: number
    readonly getPositionEmbedding: (positionIndex: number) => readonly number[]
}

const defaultSeed = 2024
const initializationScale = 0.02

/**
 * Crée une table d'embeddings de position CPU.
 *
 * Un token embedding dit "quel symbole suis-je ?".
 * Un position embedding ajoute "où suis-je dans la séquence ?".
 *
 * Dans ce module, les embeddings de position sont initialisés mais pas entraînés. Ils servent
 * à rendre visible le mécanisme sans introduire encore de gradients.
 */
export function createPositionEmbeddingTable(
    options: PositionEmbeddingTableOptions,
): PositionEmbeddingTable {
    validatePositiveInteger(options.maxSequenceLength, 'maxSequenceLength')
    validatePositiveInteger(options.embeddingDimension, 'embeddingDimension')

    const random = createDeterministicRandom(options.seed ?? defaultSeed)
    const vectors = Array.from({ length: options.maxSequenceLength }, () =>
        Array.from(
            { length: options.embeddingDimension },
            () => (random() * 2 - 1) * initializationScale,
        ),
    )

    return {
        embeddingDimension: options.embeddingDimension,
        getPositionEmbedding: (positionIndex) => readPositionEmbedding(positionIndex, vectors),
        maxSequenceLength: options.maxSequenceLength,
        vectors,
    }
}

export function getPositionEmbedding(
    positionIndex: number,
    positionTable: PositionEmbeddingTable,
): readonly number[] {
    return readPositionEmbedding(positionIndex, positionTable.vectors)
}

/**
 * Ajoute l'information de position aux token embeddings.
 *
 * Formule pédagogique:
 * représentation = tokenEmbedding + positionEmbedding
 *
 * Les deux vecteurs doivent avoir la même dimension. Sinon, l'addition n'a pas de sens:
 * on ne peut pas ajouter un vecteur de 4 valeurs avec un vecteur de 8 valeurs.
 */
export function addPositionalEmbeddings(
    tokenVectors: readonly (readonly number[])[],
    positionTable: PositionEmbeddingTable,
): readonly (readonly number[])[] {
    if (tokenVectors.length > positionTable.maxSequenceLength) {
        throw new Error(
            `La séquence contient ${String(
                tokenVectors.length,
            )} positions, mais la table de positions en supporte seulement ${String(
                positionTable.maxSequenceLength,
            )}.`,
        )
    }

    return tokenVectors.map((tokenVector, positionIndex) => {
        validateVectorDimension(tokenVector, positionTable.embeddingDimension, positionIndex)

        const positionVector = positionTable.getPositionEmbedding(positionIndex)

        return tokenVector.map((value, dimensionIndex) => {
            const positionValue = readNumberAt(positionVector, dimensionIndex)

            return value + positionValue
        })
    })
}

function createDeterministicRandom(seed: number): () => number {
    let state = normalizeSeed(seed)

    return () => {
        // Même générateur pseudo-aléatoire simple que dans les modules précédents.
        // Il n'est pas cryptographique; il sert seulement à rendre les démos reproductibles.
        state = (state * 1664525 + 1013904223) % 4294967296

        return state / 4294967296
    }
}

function readPositionEmbedding(
    positionIndex: number,
    vectors: readonly (readonly number[])[],
): readonly number[] {
    validatePositionIndex(positionIndex, vectors.length)

    const vector = vectors[positionIndex]

    if (vector === undefined) {
        throw new Error(
            `Embedding de position introuvable pour la position ${String(positionIndex)}.`,
        )
    }

    return vector
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

function validatePositionIndex(positionIndex: number, maxSequenceLength: number): void {
    if (
        !Number.isInteger(positionIndex) ||
        positionIndex < 0 ||
        positionIndex >= maxSequenceLength
    ) {
        throw new Error(
            `positionIndex doit être un entier entre 0 et ${String(
                maxSequenceLength - 1,
            )}. Valeur reçue: ${String(positionIndex)}.`,
        )
    }
}

function validateVectorDimension(
    vector: readonly number[],
    embeddingDimension: number,
    positionIndex: number,
): void {
    if (vector.length !== embeddingDimension) {
        throw new Error(
            `Le vecteur de token à la position ${String(positionIndex)} doit avoir ${String(
                embeddingDimension,
            )} dimensions. Dimension reçue: ${String(vector.length)}.`,
        )
    }
}
