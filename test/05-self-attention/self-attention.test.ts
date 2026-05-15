import { describe, expect, it } from 'vitest'

import {
    createSelfAttention,
    dotProduct,
    multiplyMatrixVector,
    softmax,
} from '../../src/modules/05-self-attention/index.js'

describe('softmax', () => {
    it('produit des probabilites qui somment a 1', () => {
        const probabilities = softmax([1, 2, 3])

        expect(probabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1)
    })

    it('favorise les scores plus eleves', () => {
        const probabilities = softmax([1, 2, 3])

        expect(readNumberAt(probabilities, 2)).toBeGreaterThan(readNumberAt(probabilities, 1))
        expect(readNumberAt(probabilities, 1)).toBeGreaterThan(readNumberAt(probabilities, 0))
    })
})

function readNumberAt(values: readonly number[], index: number): number {
    const value = values[index]

    if (value === undefined) {
        throw new Error(`Valeur introuvable a l'index ${String(index)}.`)
    }

    return value
}

describe('dotProduct', () => {
    it('calcule un produit scalaire', () => {
        expect(dotProduct([1, 2, 3], [4, 5, 6])).toBe(32)
    })
})

describe('multiplyMatrixVector', () => {
    it('multiplie une matrice par un vecteur', () => {
        expect(
            multiplyMatrixVector(
                [
                    [1, 2],
                    [3, 4],
                ],
                [5, 6],
            ),
        ).toEqual([17, 39])
    })
})

describe('createSelfAttention', () => {
    it('cree les matrices aux bonnes dimensions', () => {
        const attention = createSelfAttention({
            attentionDimension: 2,
            embeddingDimension: 3,
        })

        expect(attention.queryWeights).toHaveLength(2)
        expect(attention.queryWeights[0]).toHaveLength(3)
        expect(attention.keyWeights).toHaveLength(2)
        expect(attention.valueWeights).toHaveLength(2)
    })

    it('initialise deterministiquement avec le meme seed', () => {
        const attentionA = createSelfAttention({
            attentionDimension: 2,
            embeddingDimension: 3,
            seed: 123,
        })
        const attentionB = createSelfAttention({
            attentionDimension: 2,
            embeddingDimension: 3,
            seed: 123,
        })

        expect(attentionA.queryWeights).toEqual(attentionB.queryWeights)
        expect(attentionA.keyWeights).toEqual(attentionB.keyWeights)
        expect(attentionA.valueWeights).toEqual(attentionB.valueWeights)
    })

    it('produit des projections differentes avec des seeds differents', () => {
        const attentionA = createSelfAttention({
            attentionDimension: 2,
            embeddingDimension: 3,
            seed: 123,
        })
        const attentionB = createSelfAttention({
            attentionDimension: 2,
            embeddingDimension: 3,
            seed: 456,
        })

        expect(attentionA.queryWeights).not.toEqual(attentionB.queryWeights)
    })

    it('retourne une sortie par position', () => {
        const attention = createSelfAttention({
            attentionDimension: 2,
            embeddingDimension: 2,
            seed: 123,
        })
        const result = attention.applyCausalSelfAttention([
            [1, 0],
            [0, 1],
            [1, 1],
        ])

        expect(result.outputVectors).toHaveLength(3)
        expect(result.attentionWeights).toHaveLength(3)
    })

    it('produit des sorties de dimension attentionDimension', () => {
        const attention = createSelfAttention({
            attentionDimension: 3,
            embeddingDimension: 2,
            seed: 123,
        })
        const result = attention.applyCausalSelfAttention([
            [1, 0],
            [0, 1],
        ])

        expect(result.outputVectors[0]).toHaveLength(3)
        expect(result.outputVectors[1]).toHaveLength(3)
    })

    it('empeche la premiere position de regarder le futur', () => {
        const attention = createSelfAttention({
            attentionDimension: 2,
            embeddingDimension: 2,
            seed: 123,
        })
        const result = attention.applyCausalSelfAttention([
            [1, 0],
            [0, 1],
            [1, 1],
        ])

        expect(result.attentionScores[0]).toEqual([
            expect.any(Number),
            Number.NEGATIVE_INFINITY,
            Number.NEGATIVE_INFINITY,
        ])
        expect(result.attentionWeights[0]).toEqual([1, 0, 0])
    })

    it('rejette les dimensions invalides', () => {
        expect(() => createSelfAttention({ embeddingDimension: 0 })).toThrow(
            'embeddingDimension doit etre un entier strictement positif.',
        )
        expect(() => createSelfAttention({ attentionDimension: 0, embeddingDimension: 2 })).toThrow(
            'attentionDimension doit etre un entier strictement positif.',
        )
    })

    it("rejette les vecteurs d'entree incompatibles", () => {
        const attention = createSelfAttention({
            embeddingDimension: 2,
        })

        expect(() => attention.applyCausalSelfAttention([])).toThrow(
            'applyCausalSelfAttention attend au moins un vecteur.',
        )
        expect(() => attention.applyCausalSelfAttention([[1, 2, 3]])).toThrow(
            "Le vecteur d'entree 0 doit avoir 2 dimensions. Dimension recue: 3.",
        )
    })
})
