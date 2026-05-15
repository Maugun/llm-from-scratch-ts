import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from './index.js'

const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const rawText = await loadTextFile(corpusPath)
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)

console.info('Module 2 - Dataset loader')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le fichier texte')
console.info('2. Creer le tokenizer a partir de ce texte')
console.info('3. Encoder le texte pour creer le dataset')
console.info('')
console.info(`Fichier lu: ${corpusPath}`)
console.info('')
console.info('Contenu du fichier:')
console.info(rawText)
console.info(`Caracteres dans le texte brut: ${String(rawText.length)}`)
console.info(`Taille du vocabulaire du tokenizer: ${String(tokenizer.vocabularySize)}`)
console.info(`Tokens totaux: ${String(dataset.totalTokens)}`)
console.info(`Tokens train: ${String(dataset.trainTokenCount)}`)
console.info(`Tokens validation: ${String(dataset.validationTokenCount)}`)
console.info('')
console.info('Premiers ids du dataset:')
const firstTokenIds = dataset.tokenIds.slice(0, 24)
console.info(firstTokenIds)
console.info('')
console.info('Ces memes ids decodes:')
console.info(tokenizer.decode(firstTokenIds))
