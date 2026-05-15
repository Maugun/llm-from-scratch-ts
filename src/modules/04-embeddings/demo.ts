import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../02-dataset-loader/index.js'
import { cosineSimilarity, createEmbeddingTable } from './index.js'

const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const rawText = await loadTextFile(corpusPath)
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)
const embeddingTable = createEmbeddingTable({
    embeddingDimension: 4,
    seed: 123,
    vocabularySize: tokenizer.vocabularySize,
})
const defaultTextToEmbed = 'le'

console.info('Module 4 - Embeddings CPU')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le fichier texte')
console.info('2. Creer le tokenizer')
console.info('3. Creer le dataset de tokens')
console.info('4. Creer une table d embeddings')
console.info('5. Remplacer des token ids par des vecteurs')
console.info('')
console.info(`Fichier lu: ${corpusPath}`)
console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caracteres`)
console.info(`Dimension des embeddings: ${String(embeddingTable.embeddingDimension)}`)
console.info(
    `Chaque token est donc remplace par un vecteur de ${String(
        embeddingTable.embeddingDimension,
    )} nombres.`,
)
console.info(
    `Valeurs stockees dans la table: ${String(tokenizer.vocabularySize)} x ${String(
        embeddingTable.embeddingDimension,
    )} = ${String(tokenizer.vocabularySize * embeddingTable.embeddingDimension)} nombres.`,
)
console.info(`Tokens dans le dataset: ${String(dataset.totalTokens)}`)
console.info('')

showEmbeddingsForText(defaultTextToEmbed)

if (process.stdin.isTTY) {
    await startInteractivePrompt()
} else {
    console.info('')
    console.info(
        "Mode non interactif detecte: lance cette demo dans un terminal pour choisir d'autres lettres.",
    )
}

function showEmbeddingsForText(textToEmbed: string): void {
    const tokenIds = tokenizer.encode(textToEmbed)
    const embeddedSequence = embeddingTable.embedSequence(tokenIds)

    console.info(`Texte choisi: "${textToEmbed}"`)
    console.info('Token ids:')
    console.info(tokenIds)
    console.info('')
    console.info('Vecteurs associes:')

    for (const [index, vector] of embeddedSequence.entries()) {
        const tokenId = readTokenIdAt(tokenIds, index)
        const character = tokenizer.decode([tokenId])

        console.info(`  "${character}" -> ${formatVector(vector)}`)
    }

    if (embeddedSequence.length >= 2) {
        console.info('')
        console.info('Similarite cosinus entre les deux premiers vecteurs:')
        console.info(
            cosineSimilarity(
                readVectorAt(embeddedSequence, 0),
                readVectorAt(embeddedSequence, 1),
            ).toFixed(3),
        )
    }

    console.info('')
    console.info(
        "Ces vecteurs sont initialises, pas encore appris: la similarite n'a pas encore de sens linguistique fort.",
    )
}

async function startInteractivePrompt(): Promise<void> {
    console.info('')
    console.info('Choisis une ou plusieurs lettres du vocabulaire pour voir leurs embeddings.')
    console.info('Appuie sur ENTREE pour valider, ou sur ESC pour quitter.')
    console.info('')

    let currentInput = ''

    process.stdin.setRawMode(true)
    process.stdin.resume()
    process.stdin.setEncoding('utf8')

    await new Promise<void>((resolve) => {
        const handleInput = (input: string): void => {
            if (input === '\u001B' || input === '\u0003') {
                process.stdin.off('data', handleInput)
                process.stdin.setRawMode(false)
                process.stdin.pause()
                console.info('')
                console.info('Demo terminee.')
                resolve()

                return
            }

            if (input === '\r' || input === '\n') {
                console.info('')

                if (currentInput.length === 0) {
                    console.info('Aucun texte saisi.')
                } else {
                    try {
                        showEmbeddingsForText(currentInput)
                    } catch {
                        console.info(
                            `Le texte "${currentInput}" contient au moins un caractere absent du vocabulaire.`,
                        )
                    }
                }

                currentInput = ''
                console.info('')
                console.info('Saisis un autre texte, ou appuie sur ESC pour quitter.')

                return
            }

            if (input === '\u007F' || input === '\b') {
                currentInput = currentInput.slice(0, -1)

                return
            }

            currentInput += input
            process.stdout.write(input)
        }
        process.stdin.on('data', handleInput)
    })
}

function formatVector(vector: readonly number[]): string {
    return `[${vector.map((value) => value.toFixed(4)).join(', ')}]`
}

function readTokenIdAt(tokenIdsToRead: readonly number[], index: number): number {
    const tokenId = tokenIdsToRead[index]

    if (tokenId === undefined) {
        throw new Error(`Token introuvable a l'index ${String(index)}.`)
    }

    return tokenId
}

function readVectorAt(vectors: readonly (readonly number[])[], index: number): readonly number[] {
    const vector = vectors[index]

    if (vector === undefined) {
        throw new Error(`Vecteur introuvable a l'index ${String(index)}.`)
    }

    return vector
}
