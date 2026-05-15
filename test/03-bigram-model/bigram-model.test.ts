import { join } from 'node:path'

import { describe, expect, it } from 'vitest'

import { createCharacterTokenizer } from '../../src/modules/01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../../src/modules/02-dataset-loader/index.js'
import { createBigramModel } from '../../src/modules/03-bigram-model/index.js'

describe('createBigramModel', () => {
    it('compte les transitions sur une séquence simple', () => {
        const model = createBigramModel([0, 1, 0, 1, 2], 3)

        expect(model.getTransitionCount(0, 1)).toBe(2)
        expect(model.getTransitionCount(1, 0)).toBe(1)
        expect(model.getTransitionCount(1, 2)).toBe(1)
        expect(model.getTransitionCount(2, 0)).toBe(0)
    })

    it('expose le nombre total de transitions', () => {
        const model = createBigramModel([0, 1, 2, 1], 3)

        expect(model.totalTransitions).toBe(3)
    })

    it('normalise les probabilités pour un token donné', () => {
        const model = createBigramModel([0, 1, 0, 2, 0, 2], 3)

        expect(model.getNextTokenProbabilities(0)).toEqual([0, 1 / 3, 2 / 3])
    })

    it("retourne une distribution vide si un token n'a aucune transition sortante", () => {
        const model = createBigramModel([0, 1], 3)

        expect(model.getNextTokenProbabilities(2)).toEqual([0, 0, 0])
        expect(model.predictMostLikelyNextToken(2)).toBeUndefined()
    })

    it('prédit le token suivant le plus probable', () => {
        const model = createBigramModel([0, 1, 0, 2, 0, 2], 3)

        expect(model.predictMostLikelyNextToken(0)).toBe(2)
    })

    it("départage les égalités avec le premier token dans l'ordre des ids", () => {
        const model = createBigramModel([0, 2, 0, 1], 3)

        expect(model.getNextTokenProbabilities(0)).toEqual([0, 0.5, 0.5])
        expect(model.predictMostLikelyNextToken(0)).toBe(1)
    })

    it('rejette un vocabularySize invalide', () => {
        expect(() => createBigramModel([0, 1], 0)).toThrow(
            'vocabularySize doit être un entier strictement positif.',
        )
    })

    it('rejette un token id hors vocabulaire', () => {
        expect(() => createBigramModel([0, 3], 3)).toThrow(
            'tokenId doit être un entier entre 0 et 2. Valeur reçue: 3.',
        )

        const model = createBigramModel([0, 1], 2)

        expect(() => model.getTransitionCount(0, 2)).toThrow(
            'tokenId doit être un entier entre 0 et 1. Valeur reçue: 2.',
        )
    })

    it('fonctionne avec le tokenizer et le dataset loader', async () => {
        const rawText = await loadTextFile(join(process.cwd(), 'data', 'tiny-corpus.txt'))
        const tokenizer = createCharacterTokenizer(rawText)
        const dataset = createTokenDataset(rawText, tokenizer)
        const model = createBigramModel(dataset.trainTokenIds, tokenizer.vocabularySize)
        const bTokenId = readFirstTokenId(tokenizer.encode('b'))
        const oTokenId = readFirstTokenId(tokenizer.encode('o'))

        expect(model.vocabularySize).toBe(tokenizer.vocabularySize)
        expect(model.totalTransitions).toBe(dataset.trainTokenIds.length - 1)
        expect(model.getTransitionCount(bTokenId, oTokenId)).toBeGreaterThan(0)
    })
})

function readFirstTokenId(tokenIds: readonly number[]): number {
    const tokenId = tokenIds[0]

    if (tokenId === undefined) {
        throw new Error('Le tokenizer aurait dû produire un token.')
    }

    return tokenId
}
