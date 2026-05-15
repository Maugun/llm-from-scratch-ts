import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../02-dataset-loader/index.js'
import { createEmbeddingTable } from '../04-embeddings/index.js'
import { addPositionalEmbeddings, createPositionEmbeddingTable } from './index.js'

const maxSequenceLength = 16
const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const rawText = await loadTextFile(corpusPath)
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)
const tokenEmbeddingTable = createEmbeddingTable({
    embeddingDimension: 4,
    seed: 123,
    vocabularySize: tokenizer.vocabularySize,
})
const positionEmbeddingTable = createPositionEmbeddingTable({
    embeddingDimension: tokenEmbeddingTable.embeddingDimension,
    maxSequenceLength,
    seed: 456,
})
const defaultText = 'llm'

console.info('Module 7 - Positional encoding CPU')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le fichier texte')
console.info('2. Créer le tokenizer')
console.info('3. Créer le dataset de tokens')
console.info('4. Transformer les token ids en token embeddings')
console.info('5. Ajouter les position embeddings')
console.info('')
console.info(`Fichier lu: ${corpusPath}`)
console.info('')
console.info('Contenu du corpus:')
console.info(rawText)
console.info('')
console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caractères`)
console.info(`Tokens dans le dataset: ${String(dataset.totalTokens)}`)
console.info(`Dimension des embeddings: ${String(tokenEmbeddingTable.embeddingDimension)}`)
console.info(`Longueur maximale supportée: ${String(positionEmbeddingTable.maxSequenceLength)}`)
console.info('')
console.info('Idée clé:')
console.info('  token embedding = identité du symbole')
console.info('  position embedding = emplacement dans la séquence')
console.info('  somme des deux = symbole + emplacement')
console.info('')

showPositionalEncodingForText(defaultText)

if (process.stdin.isTTY) {
    await startInteractivePrompt()
} else {
    console.info('')
    console.info(
        "Mode non interactif détecté: lance cette démo dans un terminal pour essayer d'autres textes.",
    )
}

function showPositionalEncodingForText(text: string): void {
    const tokenIds = tokenizer.encode(text)
    const tokenVectors = tokenEmbeddingTable.embedSequence(tokenIds)
    const positionedVectors = addPositionalEmbeddings(tokenVectors, positionEmbeddingTable)

    console.info(`Texte choisi: "${text}"`)
    console.info('Token ids:')
    console.info(tokenIds)
    console.info('')

    printVectorsByPosition('Token embeddings seuls:', tokenIds, tokenVectors)
    printPositionVectors(tokenIds)
    printVectorsByPosition('Token + position embeddings:', tokenIds, positionedVectors)
    explainRepeatedTokenDifference(text, tokenIds, tokenVectors, positionedVectors)
}

async function startInteractivePrompt(): Promise<void> {
    console.info('')
    console.info('Saisis un petit texte avec les caractères du corpus.')
    console.info('Appuie sur ENTRÉE pour ajouter les positions, ou sur ESC pour quitter.')
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
                console.info('Démo terminée.')
                resolve()

                return
            }

            if (input === '\r' || input === '\n') {
                console.info('')

                if (currentInput.length === 0) {
                    console.info('Aucun texte saisi.')
                } else {
                    try {
                        showPositionalEncodingForText(currentInput)
                    } catch (error) {
                        console.info(toEducationalErrorMessage(currentInput, error))
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

function explainRepeatedTokenDifference(
    text: string,
    tokenIds: readonly number[],
    tokenVectors: readonly (readonly number[])[],
    positionedVectors: readonly (readonly number[])[],
): void {
    const firstCharacter = Array.from(text)[0]

    if (firstCharacter === undefined) {
        return
    }

    const repeatedPosition = Array.from(text).findIndex(
        (character, index) => index > 0 && character === firstCharacter,
    )

    if (repeatedPosition === -1) {
        return
    }

    const firstTokenId = readTokenIdAt(tokenIds, 0)
    const repeatedTokenId = readTokenIdAt(tokenIds, repeatedPosition)

    if (firstTokenId !== repeatedTokenId) {
        return
    }

    console.info('')
    console.info(`Comparaison des deux "${firstCharacter}":`)
    console.info('Même token embedding:')
    console.info(`  position 0 -> ${formatVector(readVectorAt(tokenVectors, 0))}`)
    console.info(
        `  position ${String(repeatedPosition)} -> ${formatVector(
            readVectorAt(tokenVectors, repeatedPosition),
        )}`,
    )
    console.info('Après ajout de la position, les vecteurs deviennent différents:')
    console.info(`  position 0 -> ${formatVector(readVectorAt(positionedVectors, 0))}`)
    console.info(
        `  position ${String(repeatedPosition)} -> ${formatVector(
            readVectorAt(positionedVectors, repeatedPosition),
        )}`,
    )
}

function formatVector(vector: readonly number[]): string {
    return `[${vector.map((value) => value.toFixed(4)).join(', ')}]`
}

function printPositionVectors(tokenIds: readonly number[]): void {
    console.info('')
    console.info('Position embeddings:')

    for (const [positionIndex] of tokenIds.entries()) {
        const positionVector = positionEmbeddingTable.getPositionEmbedding(positionIndex)

        console.info(`  position ${String(positionIndex)} -> ${formatVector(positionVector)}`)
    }

    console.info('')
}

function printVectorsByPosition(
    title: string,
    tokenIds: readonly number[],
    vectors: readonly (readonly number[])[],
): void {
    console.info(title)

    for (const [index, vector] of vectors.entries()) {
        const tokenId = readTokenIdAt(tokenIds, index)
        const character = tokenizer.decode([tokenId])

        console.info(`  position ${String(index)} "${character}" -> ${formatVector(vector)}`)
    }

    console.info('')
}

function readTokenIdAt(tokenIds: readonly number[], index: number): number {
    const tokenId = tokenIds[index]

    if (tokenId === undefined) {
        throw new Error(`Token introuvable à l'index ${String(index)}.`)
    }

    return tokenId
}

function readVectorAt(vectors: readonly (readonly number[])[], index: number): readonly number[] {
    const vector = vectors[index]

    if (vector === undefined) {
        throw new Error(`Vecteur introuvable à l'index ${String(index)}.`)
    }

    return vector
}

function toEducationalErrorMessage(text: string, error: unknown): string {
    const characterCount = Array.from(text).length

    if (characterCount > maxSequenceLength) {
        return `Le texte contient ${String(characterCount)} caractères, mais cette démo supporte au maximum ${String(
            maxSequenceLength,
        )} positions.`
    }

    if (error instanceof Error) {
        return `Impossible d'encoder ou de positionner "${text}": ${error.message}`
    }

    return `Impossible d'encoder ou de positionner "${text}".`
}
