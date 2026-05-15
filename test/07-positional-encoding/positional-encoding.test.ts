import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { createCharacterTokenizer } from '../../src/modules/01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../../src/modules/02-dataset-loader/index.js'
import { createEmbeddingTable } from '../../src/modules/04-embeddings/index.js'
import {
    addPositionalEmbeddings,
    createPositionEmbeddingTable,
    getPositionEmbedding,
} from '../../src/modules/07-positional-encoding/index.js'

describe('createPositionEmbeddingTable', () => {
    it('crée une table aux bonnes dimensions', () => {
        const table = createPositionEmbeddingTable({
            embeddingDimension: 4,
            maxSequenceLength: 3,
        })

        expect(table.vectors).toHaveLength(3)
        expect(table.vectors[0]).toHaveLength(4)
        expect(table.maxSequenceLength).toBe(3)
        expect(table.embeddingDimension).toBe(4)
    })

    it('initialise les positions déterministiquement avec le même seed', () => {
        const tableA = createPositionEmbeddingTable({
            embeddingDimension: 3,
            maxSequenceLength: 2,
            seed: 123,
        })
        const tableB = createPositionEmbeddingTable({
            embeddingDimension: 3,
            maxSequenceLength: 2,
            seed: 123,
        })

        expect(tableA.vectors).toEqual(tableB.vectors)
    })

    it('produit des positions différentes avec des seeds différents', () => {
        const tableA = createPositionEmbeddingTable({
            embeddingDimension: 3,
            maxSequenceLength: 2,
            seed: 123,
        })
        const tableB = createPositionEmbeddingTable({
            embeddingDimension: 3,
            maxSequenceLength: 2,
            seed: 456,
        })

        expect(tableA.vectors).not.toEqual(tableB.vectors)
    })

    it('retourne la bonne ligne pour une position', () => {
        const table = createPositionEmbeddingTable({
            embeddingDimension: 2,
            maxSequenceLength: 3,
            seed: 123,
        })

        expect(table.getPositionEmbedding(1)).toBe(table.vectors[1])
        expect(getPositionEmbedding(1, table)).toBe(table.vectors[1])
    })

    it('rejette une position hors table', () => {
        const table = createPositionEmbeddingTable({
            embeddingDimension: 2,
            maxSequenceLength: 3,
            seed: 123,
        })

        expect(() => getPositionEmbedding(3, table)).toThrow(
            'positionIndex doit être un entier entre 0 et 2. Valeur reçue: 3.',
        )
    })

    it('rejette les dimensions invalides', () => {
        expect(() =>
            createPositionEmbeddingTable({
                embeddingDimension: 2,
                maxSequenceLength: 0,
            }),
        ).toThrow('maxSequenceLength doit être un entier strictement positif.')

        expect(() =>
            createPositionEmbeddingTable({
                embeddingDimension: 0,
                maxSequenceLength: 2,
            }),
        ).toThrow('embeddingDimension doit être un entier strictement positif.')
    })
})

describe('addPositionalEmbeddings', () => {
    it('garde le nombre de positions', () => {
        const table = createPositionEmbeddingTable({
            embeddingDimension: 2,
            maxSequenceLength: 3,
            seed: 123,
        })
        const positionedVectors = addPositionalEmbeddings(
            [
                [1, 1],
                [2, 2],
            ],
            table,
        )

        expect(positionedVectors).toHaveLength(2)
    })

    it('retourne une séquence vide si aucun token vector n’est fourni', () => {
        const table = createPositionEmbeddingTable({
            embeddingDimension: 2,
            maxSequenceLength: 3,
            seed: 123,
        })

        expect(addPositionalEmbeddings([], table)).toEqual([])
    })

    it('garde embeddingDimension pour chaque vecteur de sortie', () => {
        const table = createPositionEmbeddingTable({
            embeddingDimension: 2,
            maxSequenceLength: 3,
            seed: 123,
        })
        const positionedVectors = addPositionalEmbeddings(
            [
                [1, 1],
                [2, 2],
            ],
            table,
        )

        expect(positionedVectors[0]).toHaveLength(2)
        expect(positionedVectors[1]).toHaveLength(2)
    })

    it('rend deux mêmes token embeddings différents à deux positions différentes', () => {
        const table = createPositionEmbeddingTable({
            embeddingDimension: 2,
            maxSequenceLength: 2,
            seed: 123,
        })
        const positionedVectors = addPositionalEmbeddings(
            [
                [1, 1],
                [1, 1],
            ],
            table,
        )

        expect(positionedVectors[0]).not.toEqual(positionedVectors[1])
    })

    it('rejette une séquence qui dépasse maxSequenceLength', () => {
        const table = createPositionEmbeddingTable({
            embeddingDimension: 2,
            maxSequenceLength: 1,
            seed: 123,
        })

        expect(() =>
            addPositionalEmbeddings(
                [
                    [1, 1],
                    [2, 2],
                ],
                table,
            ),
        ).toThrow(
            'La séquence contient 2 positions, mais la table de positions en supporte seulement 1.',
        )
    })

    it("rejette un vecteur d'entrée avec une mauvaise dimension", () => {
        const table = createPositionEmbeddingTable({
            embeddingDimension: 2,
            maxSequenceLength: 2,
            seed: 123,
        })

        expect(() => addPositionalEmbeddings([[1, 2, 3]], table)).toThrow(
            'Le vecteur de token à la position 0 doit avoir 2 dimensions. Dimension reçue: 3.',
        )
    })

    it('fonctionne avec le tokenizer, le mini corpus et les embeddings', async () => {
        const rawText = await loadTextFile(join(process.cwd(), 'data', 'tiny-corpus.txt'))
        const tokenizer = createCharacterTokenizer(rawText)
        const dataset = createTokenDataset(rawText, tokenizer)
        const tokenEmbeddingTable = createEmbeddingTable({
            embeddingDimension: 4,
            vocabularySize: tokenizer.vocabularySize,
        })
        const positionEmbeddingTable = createPositionEmbeddingTable({
            embeddingDimension: tokenEmbeddingTable.embeddingDimension,
            maxSequenceLength: 8,
        })
        const tokenVectors = tokenEmbeddingTable.embedSequence(dataset.trainTokenIds.slice(0, 3))
        const positionedVectors = addPositionalEmbeddings(tokenVectors, positionEmbeddingTable)

        expect(positionedVectors).toHaveLength(tokenVectors.length)
        expect(positionedVectors[0]).toHaveLength(tokenEmbeddingTable.embeddingDimension)
    })
})
