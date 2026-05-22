import { describe, expect, it } from 'vitest'
import { join } from 'node:path'

import {
    cleanCorpusText,
    createDefaultCleanCorpusOutputPath,
} from '../../src/tools/corpus-cleaner.js'

describe('cleanCorpusText', () => {
    it('remplace les retours à la ligne simples par des espaces', () => {
        const result = cleanCorpusText('Bonjour\nle monde.\nCeci est un test.')

        expect(result.cleanedText).toBe('Bonjour le monde. Ceci est un test.')
        expect(result.before.lineCount).toBe(3)
        expect(result.after.lineCount).toBe(1)
    })

    it('peut conserver les paragraphes séparés par une ligne vide', () => {
        const result = cleanCorpusText('Premier paragraphe.\n\nDeuxième paragraphe.', {
            keepParagraphs: true,
        })

        expect(result.cleanedText).toBe('Premier paragraphe.\n\nDeuxième paragraphe.')
        expect(result.after.lineCount).toBe(3)
    })

    it('peut recoller les mots coupés par une césure de fin de ligne', () => {
        const result = cleanCorpusText('La ma-\ngie arrive.', {
            fixHyphenation: true,
        })

        expect(result.cleanedText).toBe('La magie arrive.')
    })

    it('nettoie les espaces inutiles avant la ponctuation', () => {
        const result = cleanCorpusText('Bonjour   , le monde   !')

        expect(result.cleanedText).toBe('Bonjour, le monde!')
    })
})

describe('createDefaultCleanCorpusOutputPath', () => {
    it('ajoute .clean avant l’extension', () => {
        expect(createDefaultCleanCorpusOutputPath('data/private/text.txt')).toBe(
            join('data', 'private', 'text.clean.txt'),
        )
    })

    it('ajoute .clean à la fin si le chemin n’a pas d’extension', () => {
        expect(createDefaultCleanCorpusOutputPath('data/private/text')).toBe(
            join('data', 'private', 'text.clean'),
        )
    })
})
