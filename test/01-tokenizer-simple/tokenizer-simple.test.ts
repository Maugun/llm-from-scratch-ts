import { describe, expect, it } from 'vitest'

import { createCharacterTokenizer } from '../../src/modules/01-tokenizer-simple/index.js'

describe('createCharacterTokenizer', () => {
    it('construit un vocabulaire trié et stable depuis un texte simple', () => {
        const tokenizer = createCharacterTokenizer('banana')

        expect(tokenizer.vocabulary).toEqual(['a', 'b', 'n'])
        expect([...tokenizer.charToId.entries()]).toEqual([
            ['a', 0],
            ['b', 1],
            ['n', 2],
        ])
        expect([...tokenizer.idToChar.entries()]).toEqual([
            [0, 'a'],
            [1, 'b'],
            [2, 'n'],
        ])
    })

    it('encode puis décode un texte connu sans perte', () => {
        const tokenizer = createCharacterTokenizer('hello world')
        const tokenIds = tokenizer.encode('hello')

        expect(tokenIds).toEqual([3, 2, 4, 4, 5])
        expect(tokenizer.decode(tokenIds)).toBe('hello')
    })

    it('expose correctement la taille du vocabulaire', () => {
        const tokenizer = createCharacterTokenizer('aabbcc')

        expect(tokenizer.vocabularySize).toBe(3)
    })

    it("échoue clairement lorsqu'un caractère est absent du vocabulaire", () => {
        const tokenizer = createCharacterTokenizer('abc')

        expect(() => tokenizer.encode('abcd')).toThrow(
            'Impossible d\'encoder le caractère inconnu "d".',
        )
    })

    it("échoue clairement lorsqu'un id est absent du vocabulaire", () => {
        const tokenizer = createCharacterTokenizer('abc')

        expect(() => tokenizer.decode([0, 3])).toThrow("Impossible de décoder l'id inconnu 3.")
    })

    it('traite les espaces et la ponctuation comme des caractères normaux', () => {
        const tokenizer = createCharacterTokenizer('Salut, LLM !')
        const text = 'LLM !'

        expect(tokenizer.decode(tokenizer.encode(text))).toBe(text)
        expect(tokenizer.vocabulary).toContain(' ')
        expect(tokenizer.vocabulary).toContain('!')
        expect(tokenizer.vocabulary).toContain(',')
    })

    it('documente le comportement avec un texte de référence vide', () => {
        const tokenizer = createCharacterTokenizer('')

        expect(tokenizer.vocabulary).toEqual([])
        expect(tokenizer.vocabularySize).toBe(0)
        expect(tokenizer.decode([])).toBe('')
        expect(() => tokenizer.encode('a')).toThrow(
            'Impossible d\'encoder le caractère inconnu "a".',
        )
    })
})
