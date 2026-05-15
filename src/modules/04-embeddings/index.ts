export type EmbeddingTableOptions = {
    /**
     * Nombre de tokens connus.
     *
     * La table aura une ligne par token id.
     */
    readonly vocabularySize: number

    /**
     * Nombre de dimensions dans chaque vecteur.
     *
     * Une dimension plus grande donne plus de capacité expressive, mais consomme plus de RAM.
     */
    readonly embeddingDimension: number

    /**
     * Graine optionnelle pour obtenir la même initialisation à chaque exécution.
     */
    readonly seed?: number
}

export type EmbeddingTable = {
    /**
     * Matrice des embeddings.
     *
     * vectors[tokenId] est le vecteur associé à ce token.
     */
    readonly vectors: readonly (readonly number[])[]
    readonly vocabularySize: number
    readonly embeddingDimension: number
    readonly getEmbedding: (tokenId: number) => readonly number[]
    readonly embedSequence: (tokenIds: readonly number[]) => readonly (readonly number[])[]
}

const defaultSeed = 42
const initializationScale = 0.02

/**
 * Crée une table d'embeddings CPU.
 *
 * Un token id seul est un identifiant arbitraire: le nombre 12 n'est pas "plus proche" de 13
 * que de 2 par nature. Une table d'embeddings associe donc chaque id à un vecteur de nombres.
 *
 * Mémoire / VRAM:
 * ce module utilise seulement une matrice number[][] en RAM CPU. Il ne crée aucun tenseur et
 * n'utilise pas le GPU. La VRAM consommée est donc 0.
 */
export function createEmbeddingTable(options: EmbeddingTableOptions): EmbeddingTable {
    validatePositiveInteger(options.vocabularySize, 'vocabularySize')
    validatePositiveInteger(options.embeddingDimension, 'embeddingDimension')

    const random = createDeterministicRandom(options.seed ?? defaultSeed)
    const vectors = Array.from({ length: options.vocabularySize }, () =>
        Array.from(
            { length: options.embeddingDimension },
            () => (random() * 2 - 1) * initializationScale,
        ),
    )

    return {
        embedSequence: (tokenIds) => embedSequence(tokenIds, vectors),
        embeddingDimension: options.embeddingDimension,
        getEmbedding: (tokenId) => getEmbedding(tokenId, vectors),
        vectors,
        vocabularySize: options.vocabularySize,
    }
}

/**
 * Mesure la similarité cosinus entre deux vecteurs.
 *
 * C'est utile pour inspecter des embeddings, mais ce n'est pas une preuve que le modèle
 * comprend le sens des tokens. Ici les vecteurs ne sont pas encore appris: ils sont seulement
 * initialisés de façon déterministe.
 */
export function cosineSimilarity(vectorA: readonly number[], vectorB: readonly number[]): number {
    validateSameVectorDimension(vectorA, vectorB)

    // Norme euclidienne: longueur géométrique d'un vecteur.
    // Exemple: pour [3, 4], la norme vaut sqrt(3^2 + 4^2) = 5.
    // On en a besoin car la similarité cosinus compare surtout la direction des vecteurs,
    // pas leur taille brute.
    const normA = Math.sqrt(vectorA.reduce((sum, value) => sum + value * value, 0))
    const normB = Math.sqrt(vectorB.reduce((sum, value) => sum + value * value, 0))

    if (normA === 0 || normB === 0) {
        throw new Error('cosineSimilarity ne peut pas comparer un vecteur nul.')
    }

    // Produit scalaire: somme des multiplications dimension par dimension.
    // Plus deux vecteurs pointent dans une direction proche, plus ce produit est élevé.
    // La division par les normes donne une valeur entre -1 et 1:
    // 1 = même direction, 0 = directions orthogonales, -1 = directions opposées.
    const dotProduct = vectorA.reduce((sum, value, index) => {
        const valueB = readNumberAt(vectorB, index)

        return sum + value * valueB
    }, 0)

    return dotProduct / (normA * normB)
}

function createDeterministicRandom(seed: number): () => number {
    let state = normalizeSeed(seed)

    return () => {
        // Générateur congruentiel linéaire (LCG):
        // nouveauState = (ancienState * a + c) mod m.
        //
        // Les constantes 1664525, 1013904223 et 2^32 sont des constantes classiques pour
        // produire une suite pseudo-aléatoire simple sur 32 bits. Ce n'est pas adapté à la
        // sécurité, mais c'est parfait ici: avec la même seed, on obtient toujours les mêmes
        // embeddings, donc les démos et les tests restent reproductibles.
        state = (state * 1664525 + 1013904223) % 4294967296

        // On ramène l'entier 32 bits dans l'intervalle [0, 1), comme Math.random().
        return state / 4294967296
    }
}

function embedSequence(
    tokenIds: readonly number[],
    vectors: readonly (readonly number[])[],
): readonly (readonly number[])[] {
    return tokenIds.map((tokenId) => getEmbedding(tokenId, vectors))
}

function getEmbedding(tokenId: number, vectors: readonly (readonly number[])[]): readonly number[] {
    validateTokenId(tokenId, vectors.length)

    const vector = vectors[tokenId]

    if (vector === undefined) {
        throw new Error(`Embedding introuvable pour le token ${String(tokenId)}.`)
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

function validateSameVectorDimension(vectorA: readonly number[], vectorB: readonly number[]): void {
    if (vectorA.length !== vectorB.length) {
        throw new Error(
            `Les deux vecteurs doivent avoir la même dimension. Dimensions reçues: ${String(
                vectorA.length,
            )} et ${String(vectorB.length)}.`,
        )
    }
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
