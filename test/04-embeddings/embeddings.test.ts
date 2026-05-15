import { describe, expect, it } from 'vitest'

import { cosineSimilarity, createEmbeddingTable } from '../../src/modules/04-embeddings/index.js'

describe('createEmbeddingTable', () => {
    it('crée une table aux bonnes dimensions', () => {
        const table = createEmbeddingTable({
            embeddingDimension: 4,
            vocabularySize: 3,
        })

        expect(table.vectors).toHaveLength(3)
        expect(table.vectors[0]).toHaveLength(4)
        expect(table.vocabularySize).toBe(3)
        expect(table.embeddingDimension).toBe(4)
    })

    it("s'initialise déterministiquement avec le même seed", () => {
        const tableA = createEmbeddingTable({
            embeddingDimension: 3,
            seed: 123,
            vocabularySize: 2,
        })
        const tableB = createEmbeddingTable({
            embeddingDimension: 3,
            seed: 123,
            vocabularySize: 2,
        })

        expect(tableA.vectors).toEqual(tableB.vectors)
    })

    it('produit des valeurs différentes avec des seeds différents', () => {
        const tableA = createEmbeddingTable({
            embeddingDimension: 3,
            seed: 123,
            vocabularySize: 2,
        })
        const tableB = createEmbeddingTable({
            embeddingDimension: 3,
            seed: 456,
            vocabularySize: 2,
        })

        expect(tableA.vectors).not.toEqual(tableB.vectors)
    })

    it('retourne la bonne ligne pour un token id', () => {
        const table = createEmbeddingTable({
            embeddingDimension: 2,
            seed: 123,
            vocabularySize: 3,
        })

        expect(table.getEmbedding(1)).toBe(table.vectors[1])
    })

    it('transforme une séquence de token ids en séquence de vecteurs', () => {
        const table = createEmbeddingTable({
            embeddingDimension: 2,
            seed: 123,
            vocabularySize: 3,
        })

        expect(table.embedSequence([2, 0])).toEqual([table.vectors[2], table.vectors[0]])
    })

    it('rejette un vocabularySize invalide', () => {
        expect(() =>
            createEmbeddingTable({
                embeddingDimension: 2,
                vocabularySize: 0,
            }),
        ).toThrow('vocabularySize doit être un entier strictement positif.')
    })

    it('rejette une embeddingDimension invalide', () => {
        expect(() =>
            createEmbeddingTable({
                embeddingDimension: 0,
                vocabularySize: 2,
            }),
        ).toThrow('embeddingDimension doit être un entier strictement positif.')
    })

    it('rejette un token id hors vocabulaire', () => {
        const table = createEmbeddingTable({
            embeddingDimension: 2,
            vocabularySize: 2,
        })

        expect(() => table.getEmbedding(2)).toThrow(
            'tokenId doit être un entier entre 0 et 1. Valeur reçue: 2.',
        )
        expect(() => table.embedSequence([0, 2])).toThrow(
            'tokenId doit être un entier entre 0 et 1. Valeur reçue: 2.',
        )
    })
})

describe('cosineSimilarity', () => {
    it('retourne 1 pour deux vecteurs identiques simples', () => {
        expect(cosineSimilarity([1, 2], [1, 2])).toBeCloseTo(1)
    })

    it('retourne 0 pour deux vecteurs orthogonaux simples', () => {
        expect(cosineSimilarity([1, 0], [0, 1])).toBe(0)
    })

    it("rejette deux vecteurs qui n'ont pas la même dimension", () => {
        expect(() => cosineSimilarity([1, 2], [1])).toThrow(
            'Les deux vecteurs doivent avoir la même dimension.',
        )
    })

    it('rejette un vecteur nul', () => {
        expect(() => cosineSimilarity([0, 0], [1, 0])).toThrow(
            'cosineSimilarity ne peut pas comparer un vecteur nul.',
        )
    })
})
