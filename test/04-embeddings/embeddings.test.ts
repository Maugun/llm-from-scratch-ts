import { describe, expect, it } from 'vitest'

import { cosineSimilarity, createEmbeddingTable } from '../../src/modules/04-embeddings/index.js'

describe('createEmbeddingTable', () => {
    it('cree une table aux bonnes dimensions', () => {
        const table = createEmbeddingTable({
            embeddingDimension: 4,
            vocabularySize: 3,
        })

        expect(table.vectors).toHaveLength(3)
        expect(table.vectors[0]).toHaveLength(4)
        expect(table.vocabularySize).toBe(3)
        expect(table.embeddingDimension).toBe(4)
    })

    it('initialise deterministiquement avec le meme seed', () => {
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

    it('produit des valeurs differentes avec des seeds differents', () => {
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

    it('transforme une sequence de token ids en sequence de vecteurs', () => {
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
        ).toThrow('vocabularySize doit etre un entier strictement positif.')
    })

    it('rejette une embeddingDimension invalide', () => {
        expect(() =>
            createEmbeddingTable({
                embeddingDimension: 0,
                vocabularySize: 2,
            }),
        ).toThrow('embeddingDimension doit etre un entier strictement positif.')
    })

    it('rejette un token id hors vocabulaire', () => {
        const table = createEmbeddingTable({
            embeddingDimension: 2,
            vocabularySize: 2,
        })

        expect(() => table.getEmbedding(2)).toThrow(
            'tokenId doit etre un entier entre 0 et 1. Valeur recue: 2.',
        )
        expect(() => table.embedSequence([0, 2])).toThrow(
            'tokenId doit etre un entier entre 0 et 1. Valeur recue: 2.',
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

    it("rejette deux vecteurs qui n'ont pas la meme dimension", () => {
        expect(() => cosineSimilarity([1, 2], [1])).toThrow(
            'Les deux vecteurs doivent avoir la meme dimension.',
        )
    })

    it('rejette un vecteur nul', () => {
        expect(() => cosineSimilarity([0, 0], [1, 0])).toThrow(
            'cosineSimilarity ne peut pas comparer un vecteur nul.',
        )
    })
})
