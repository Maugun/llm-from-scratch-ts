import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../02-dataset-loader/index.js'
import { createEmbeddingTable } from '../04-embeddings/index.js'
import { createTransformerBlock } from './index.js'

const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const rawText = await loadTextFile(corpusPath)
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)
const embeddingTable = createEmbeddingTable({
    embeddingDimension: 4,
    seed: 123,
    vocabularySize: tokenizer.vocabularySize,
})
const transformerBlock = createTransformerBlock({
    attentionDimension: 4,
    embeddingDimension: embeddingTable.embeddingDimension,
    feedForwardDimension: 8,
    seed: 456,
})
const defaultText = 'llm'

console.info('Module 6 - Transformer block CPU')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le fichier texte')
console.info('2. Créer le tokenizer')
console.info('3. Créer le dataset de tokens')
console.info('4. Transformer des token ids en embeddings')
console.info('5. Appliquer un bloc Transformer: LayerNorm, attention, résiduel, feed-forward')
console.info('')
console.info(`Fichier lu: ${corpusPath}`)
console.info('')
console.info('Corpus utilisé:')
console.info(rawText)
console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caractères`)
console.info(`Tokens dans le dataset: ${String(dataset.totalTokens)}`)
console.info(`Dimension des embeddings: ${String(embeddingTable.embeddingDimension)}`)
console.info(`Dimension interne de l'attention: ${String(transformerBlock.attentionDimension)}`)
console.info(`Dimension cachée du feed-forward: ${String(transformerBlock.feedForwardDimension)}`)
console.info('')
console.info('Idée clé:')
console.info("  attention = communication entre positions d'une séquence")
console.info('  résiduel = vecteur entrant de la sous-brique + correction')
console.info('  feed-forward = transformation locale de chaque position')
console.info('  layer norm = remise à l’échelle pour stabiliser les calculs')
console.info('')

showTransformerBlockForText(defaultText)

if (process.stdin.isTTY) {
    await startInteractivePrompt()
} else {
    console.info('')
    console.info(
        "Mode non interactif détecté: lance cette démo dans un terminal pour essayer d'autres textes.",
    )
}

function showTransformerBlockForText(text: string): void {
    const tokenIds = tokenizer.encode(text)
    const inputVectors = embeddingTable.embedSequence(tokenIds)
    const result = transformerBlock.applyTransformerBlock(inputVectors)

    console.info(`Texte choisi: "${text}"`)
    console.info('Token ids:')
    console.info(tokenIds)
    console.info('')

    printVectorsByPosition('Embeddings d’entrée:', tokenIds, inputVectors)

    console.info('')
    console.info('Après attention + résiduel:')
    console.info(
        "Ici, le résiduel est l'embedding d'entrée: attentionResidual = embedding + correctionAttention.",
    )
    printVectorsByPosition(
        'Correction proposée par l’attention:',
        tokenIds,
        result.attentionOutputVectors,
    )
    printVectorsByPosition('Résultat intermédiaire:', tokenIds, result.attentionResidualVectors)

    console.info('')
    console.info('Après feed-forward + résiduel:')
    console.info(
        'Ici, le résiduel est le résultat intermédiaire: sortieFinale = attentionResidual + correctionFeedForward.',
    )
    printVectorsByPosition(
        'Correction proposée par le feed-forward:',
        tokenIds,
        result.feedForwardOutputVectors,
    )
    printVectorsByPosition('Sortie finale du bloc:', tokenIds, result.outputVectors)

    console.info('')
    console.info("Poids d'attention observés:")

    for (const [position, weights] of result.attentionWeights.entries()) {
        const tokenId = readTokenIdAt(tokenIds, position)
        const character = tokenizer.decode([tokenId])
        const readableWeights = weights.map((weight, index) => {
            const targetTokenId = readTokenIdAt(tokenIds, index)
            const targetCharacter = tokenizer.decode([targetTokenId])

            return `"${targetCharacter}": ${weight.toFixed(3)}`
        })

        console.info(
            `  position ${String(position)} "${character}" regarde -> ${readableWeights.join(', ')}`,
        )
    }
}

async function startInteractivePrompt(): Promise<void> {
    console.info('')
    console.info('Saisis un petit texte avec les caractères du corpus.')
    console.info('Appuie sur ENTRÉE pour appliquer le bloc, ou sur ESC pour quitter.')
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
                        showTransformerBlockForText(currentInput)
                    } catch {
                        console.info(
                            `Le texte "${currentInput}" contient au moins un caractère absent du vocabulaire.`,
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
}

function readTokenIdAt(tokenIds: readonly number[], index: number): number {
    const tokenId = tokenIds[index]

    if (tokenId === undefined) {
        throw new Error(`Token introuvable à l'index ${String(index)}.`)
    }

    return tokenId
}
