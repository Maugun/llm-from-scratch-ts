import { mkdir, readFile, writeFile } from 'node:fs/promises'
import { dirname } from 'node:path'

export type BpeMerge = {
    readonly left: string
    readonly right: string
    readonly merged: string
}

export type BpeTokenizer = {
    readonly type: 'pedagogical-bpe'
    readonly version: 1
    readonly vocabulary: readonly string[]
    readonly vocabularySize: number
    readonly merges: readonly BpeMerge[]
    readonly tokenToId: ReadonlyMap<string, number>
    readonly idToToken: ReadonlyMap<number, string>
    readonly encode: (text: string) => number[]
    readonly decode: (tokenIds: readonly number[]) => string
}

export type BpeTokenizerTrainingOptions = {
    readonly vocabularySize: number
    readonly maxTrainingCharacters?: number
    readonly minimumMergeCount?: number
    readonly maximumMergedTokenLength?: number
    readonly maximumSpacesInMergedToken?: number
    readonly onProgress?: (progress: BpeTokenizerTrainingProgress) => void
}

type BpeMergeConstraints = {
    readonly maximumMergedTokenLength: number
    readonly maximumSpacesInMergedToken: number
}

export type BpeTokenizerTrainingProgress = {
    readonly vocabularySize: number
    readonly targetVocabularySize: number
    readonly mergeCount: number
    readonly progressRatio: number
    readonly latestMerge: BpeMerge | undefined
    readonly elapsedMs: number
}

const tokenizerVersion = 1
const unknownCharacterErrorPrefix = 'Caractère absent du vocabulaire BPE'
const defaultMinimumMergeCount = 2
const defaultMaximumMergedTokenLength = 24
const defaultMaximumSpacesInMergedToken = 1

export function trainBpeTokenizer(
    text: string,
    options: BpeTokenizerTrainingOptions,
): BpeTokenizer {
    validatePositiveInteger(options.vocabularySize, 'vocabularySize')
    const minimumMergeCount = options.minimumMergeCount ?? defaultMinimumMergeCount
    const constraints = {
        maximumMergedTokenLength:
            options.maximumMergedTokenLength ?? defaultMaximumMergedTokenLength,
        maximumSpacesInMergedToken:
            options.maximumSpacesInMergedToken ?? defaultMaximumSpacesInMergedToken,
    }

    validatePositiveInteger(minimumMergeCount, 'minimumMergeCount')
    validatePositiveInteger(constraints.maximumMergedTokenLength, 'maximumMergedTokenLength')
    validateNonNegativeInteger(constraints.maximumSpacesInMergedToken, 'maximumSpacesInMergedToken')

    const trainingText =
        options.maxTrainingCharacters === undefined
            ? text
            : Array.from(text).slice(0, options.maxTrainingCharacters).join('')

    if (trainingText.length === 0) {
        throw new Error('Le texte d’entraînement BPE ne doit pas être vide.')
    }

    // Le vocabulaire de base doit couvrir tout le corpus, même si les merges BPE
    // sont appris sur une portion limitée. Sinon un caractère rare placé tard dans
    // le fichier pourrait être impossible à encoder ensuite.
    const initialVocabulary = Array.from(new Set(Array.from(text))).sort()

    if (options.vocabularySize < initialVocabulary.length) {
        throw new Error(
            `vocabularySize doit être au moins égal au nombre de caractères uniques (${String(
                initialVocabulary.length,
            )}). Valeur reçue: ${String(options.vocabularySize)}.`,
        )
    }

    let pieces = Array.from(trainingText)
    const vocabulary = [...initialVocabulary]
    const vocabularySet = new Set(vocabulary)
    const merges: BpeMerge[] = []
    const startedAt = Date.now()

    while (vocabulary.length < options.vocabularySize) {
        const bestPair = findMostFrequentPair(pieces, constraints)

        if (bestPair === undefined || bestPair.count < minimumMergeCount) {
            break
        }

        const merged = `${bestPair.left}${bestPair.right}`

        if (vocabularySet.has(merged)) {
            break
        }

        const merge = {
            left: bestPair.left,
            merged,
            right: bestPair.right,
        }

        pieces = mergePair(pieces, bestPair.left, bestPair.right, merged)
        vocabulary.push(merged)
        vocabularySet.add(merged)
        merges.push(merge)
        options.onProgress?.({
            elapsedMs: Date.now() - startedAt,
            latestMerge: merge,
            mergeCount: merges.length,
            progressRatio: vocabulary.length / options.vocabularySize,
            targetVocabularySize: options.vocabularySize,
            vocabularySize: vocabulary.length,
        })
    }

    return createBpeTokenizer(vocabulary, merges)
}

export function encodeWithBpe(tokenizer: BpeTokenizer, text: string): number[] {
    let pieces = Array.from(text)

    for (const [characterIndex, character] of pieces.entries()) {
        if (!tokenizer.tokenToId.has(character)) {
            throw new Error(
                `${unknownCharacterErrorPrefix}: "${character}" à la position ${String(
                    characterIndex,
                )}. Le tokenizer doit être entraîné sur un corpus qui contient ce caractère.`,
            )
        }
    }

    for (const merge of tokenizer.merges) {
        pieces = mergePair(pieces, merge.left, merge.right, merge.merged)
    }

    return pieces.map((piece) => {
        const tokenId = tokenizer.tokenToId.get(piece)

        if (tokenId === undefined) {
            throw new Error(`Token BPE introuvable dans le vocabulaire: "${piece}".`)
        }

        return tokenId
    })
}

export function decodeWithBpe(tokenizer: BpeTokenizer, tokenIds: readonly number[]): string {
    return tokenIds
        .map((tokenId) => {
            const token = tokenizer.idToToken.get(tokenId)

            if (token === undefined) {
                throw new Error(`Token id BPE inconnu: ${String(tokenId)}.`)
            }

            return token
        })
        .join('')
}

export async function saveBpeTokenizer(tokenizer: BpeTokenizer, filePath: string): Promise<void> {
    await mkdir(dirname(filePath), { recursive: true })
    await writeFile(
        filePath,
        `${JSON.stringify(
            {
                merges: tokenizer.merges,
                type: tokenizer.type,
                version: tokenizer.version,
                vocabulary: tokenizer.vocabulary,
            },
            null,
            2,
        )}\n`,
        'utf8',
    )
}

export async function loadBpeTokenizer(filePath: string): Promise<BpeTokenizer> {
    const rawJson = await readFile(filePath, 'utf8')
    const parsedValue = JSON.parse(rawJson) as unknown

    return validateSerializedTokenizer(parsedValue)
}

function createBpeTokenizer(
    vocabulary: readonly string[],
    merges: readonly BpeMerge[],
): BpeTokenizer {
    const tokenToId = new Map(vocabulary.map((token, tokenId) => [token, tokenId]))
    const idToToken = new Map(vocabulary.map((token, tokenId) => [tokenId, token]))

    return {
        decode: (tokenIds) => decodeWithBpe(createBpeTokenizer(vocabulary, merges), tokenIds),
        encode: (text) => encodeWithBpe(createBpeTokenizer(vocabulary, merges), text),
        idToToken,
        merges: [...merges],
        tokenToId,
        type: 'pedagogical-bpe',
        version: tokenizerVersion,
        vocabulary: [...vocabulary],
        vocabularySize: vocabulary.length,
    }
}

function findMostFrequentPair(
    pieces: readonly string[],
    constraints: BpeMergeConstraints,
): { readonly left: string; readonly right: string; readonly count: number } | undefined {
    const counts = new Map<string, { left: string; right: string; count: number }>()

    for (let index = 0; index < pieces.length - 1; index++) {
        const left = pieces[index]
        const right = pieces[index + 1]

        if (left === undefined || right === undefined) {
            throw new Error('Paire BPE invalide pendant l’entraînement.')
        }

        if (!isAllowedMergedToken(`${left}${right}`, constraints)) {
            continue
        }

        const key = `${left}\u0000${right}`
        const existing = counts.get(key)

        if (existing === undefined) {
            counts.set(key, { count: 1, left, right })
        } else {
            counts.set(key, { ...existing, count: existing.count + 1 })
        }
    }

    return [...counts.values()].sort((left, right) => {
        if (right.count !== left.count) {
            return right.count - left.count
        }

        const leftKey = `${left.left}${left.right}`
        const rightKey = `${right.left}${right.right}`

        return leftKey.localeCompare(rightKey)
    })[0]
}

function isAllowedMergedToken(merged: string, constraints: BpeMergeConstraints): boolean {
    return (
        Array.from(merged).length <= constraints.maximumMergedTokenLength &&
        countWhitespaceCharacters(merged) <= constraints.maximumSpacesInMergedToken
    )
}

function countWhitespaceCharacters(value: string): number {
    return value.match(/\s/gu)?.length ?? 0
}

function mergePair(
    pieces: readonly string[],
    left: string,
    right: string,
    merged: string,
): string[] {
    const result: string[] = []

    for (let index = 0; index < pieces.length; index++) {
        const current = pieces[index]
        const next = pieces[index + 1]

        if (current === left && next === right) {
            result.push(merged)
            index++
        } else if (current !== undefined) {
            result.push(current)
        }
    }

    return result
}

function validateSerializedTokenizer(value: unknown): BpeTokenizer {
    if (!isRecord(value)) {
        throw new Error('Le tokenizer BPE sauvegardé doit être un objet JSON.')
    }

    if (value.type !== 'pedagogical-bpe' || value.version !== tokenizerVersion) {
        throw new Error('Version de tokenizer BPE invalide.')
    }

    if (
        !Array.isArray(value.vocabulary) ||
        value.vocabulary.some((item) => typeof item !== 'string')
    ) {
        throw new Error('vocabulary doit être un tableau de chaînes.')
    }

    if (!Array.isArray(value.merges)) {
        throw new Error('merges doit être un tableau.')
    }

    const merges = value.merges.map((merge) => {
        if (!isRecord(merge)) {
            throw new Error('Chaque merge BPE doit être un objet.')
        }

        assertString(merge.left, 'merge.left')
        assertString(merge.right, 'merge.right')
        assertString(merge.merged, 'merge.merged')

        return {
            left: merge.left,
            merged: merge.merged,
            right: merge.right,
        }
    })

    return createBpeTokenizer(value.vocabulary, merges)
}

function assertString(value: unknown, name: string): asserts value is string {
    if (typeof value !== 'string') {
        throw new Error(`${name} doit être une chaîne.`)
    }
}

function validatePositiveInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value <= 0) {
        throw new Error(`${name} doit être un entier strictement positif.`)
    }
}

function validateNonNegativeInteger(value: number, name: string): void {
    if (!Number.isInteger(value) || value < 0) {
        throw new Error(`${name} doit être un entier positif ou nul.`)
    }
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null
}
