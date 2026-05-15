import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { createTokenDataset, loadTextFile } from '../02-dataset-loader/index.js'
import { createEmbeddingTable } from '../04-embeddings/index.js'
import { createSelfAttention } from './index.js'

const corpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const rawText = await loadTextFile(corpusPath)
const tokenizer = createCharacterTokenizer(rawText)
const dataset = createTokenDataset(rawText, tokenizer)
const embeddingTable = createEmbeddingTable({
    embeddingDimension: 4,
    seed: 123,
    vocabularySize: tokenizer.vocabularySize,
})
const attention = createSelfAttention({
    attentionDimension: 4,
    embeddingDimension: embeddingTable.embeddingDimension,
    seed: 456,
})
const defaultText = 'llm'

console.info('Module 5 - Self-attention causale CPU')
console.info('')
console.info('Pipeline:')
console.info('1. Lire le fichier texte')
console.info('2. Creer le tokenizer')
console.info('3. Creer le dataset de tokens')
console.info('4. Transformer des token ids en embeddings')
console.info('5. Appliquer la self-attention causale')
console.info('')
console.info(`Fichier lu: ${corpusPath}`)
console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caracteres`)
console.info(`Tokens dans le dataset: ${String(dataset.totalTokens)}`)
console.info(`Dimension des embeddings: ${String(embeddingTable.embeddingDimension)}`)
console.info(`Dimension interne de l'attention: ${String(attention.attentionDimension)}`)
console.info('')
console.info('Pourquoi causal ? Une position peut regarder le passe et elle-meme, jamais le futur.')
console.info('Les poids affiches plus bas indiquent combien chaque position regarde les autres.')
console.info(
    'Les vecteurs de sortie sont donc contextualises: ils melangent les values autorisees.',
)
console.info('')

showAttentionForText(defaultText)

if (process.stdin.isTTY) {
    await startInteractivePrompt()
} else {
    console.info('')
    console.info(
        "Mode non interactif detecte: lance cette demo dans un terminal pour essayer d'autres textes.",
    )
}

function showAttentionForText(text: string): void {
    const tokenIds = tokenizer.encode(text)
    const inputVectors = embeddingTable.embedSequence(tokenIds)
    const result = attention.applyCausalSelfAttention(inputVectors)
    const queries = inputVectors.map((vector) =>
        multiplyMatrixVectorForDemo(attention.queryWeights, vector),
    )
    const keys = inputVectors.map((vector) =>
        multiplyMatrixVectorForDemo(attention.keyWeights, vector),
    )
    const values = inputVectors.map((vector) =>
        multiplyMatrixVectorForDemo(attention.valueWeights, vector),
    )

    console.info(`Texte choisi: "${text}"`)
    console.info('Token ids:')
    console.info(tokenIds)
    console.info('')
    console.info('Embeddings de depart, avant attention:')

    for (const [index, vector] of inputVectors.entries()) {
        const tokenId = readTokenIdAt(tokenIds, index)
        const character = tokenizer.decode([tokenId])

        console.info(`  position ${String(index)} "${character}" -> ${formatVector(vector)}`)
    }

    console.info('')
    console.info('Projections Q/K/V:')
    console.info('Q = ce que la position cherche')
    console.info('K = ce que la position annonce pour etre retrouvee')
    console.info("V = l'information qui sera vraiment melangee dans la sortie")

    for (const [index, query] of queries.entries()) {
        const tokenId = readTokenIdAt(tokenIds, index)
        const character = tokenizer.decode([tokenId])
        const key = readVectorAt(keys, index)
        const value = readVectorAt(values, index)

        console.info(`  position ${String(index)} "${character}"`)
        console.info(`    Q ${formatVector(query)}`)
        console.info(`    K ${formatVector(key)}`)
        console.info(`    V ${formatVector(value)}`)
    }

    console.info('')
    console.info('Poids d attention par position:')
    console.info('Chaque ligne se lit: cette position construit sa sortie en regardant ces tokens.')

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

    console.info('')
    console.info('Vecteurs contextualises produits:')
    console.info(
        'Ils ont la meme longueur que les values, mais ils contiennent maintenant du contexte.',
    )

    for (const [position, vector] of result.outputVectors.entries()) {
        const tokenId = readTokenIdAt(tokenIds, position)
        const character = tokenizer.decode([tokenId])

        console.info(`  position ${String(position)} "${character}" -> ${formatVector(vector)}`)
    }

    console.info('')
    console.info('Chaque sortie est une somme ponderee des values autorisees par le masque causal.')
}

async function startInteractivePrompt(): Promise<void> {
    console.info('')
    console.info('Saisis un petit texte avec les caracteres du corpus.')
    console.info('Appuie sur ENTREE pour appliquer l attention, ou sur ESC pour quitter.')
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
                        showAttentionForText(currentInput)
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

function multiplyMatrixVectorForDemo(
    matrix: readonly (readonly number[])[],
    vector: readonly number[],
): number[] {
    return matrix.map((row) =>
        row.reduce((sum, value, index) => sum + value * readNumberAt(vector, index), 0),
    )
}

function readNumberAt(values: readonly number[], index: number): number {
    const value = values[index]

    if (value === undefined) {
        throw new Error(`Valeur introuvable a l'index ${String(index)}.`)
    }

    return value
}

function readTokenIdAt(tokenIds: readonly number[], index: number): number {
    const tokenId = tokenIds[index]

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
