import { readFile, writeFile } from 'node:fs/promises'
import { dirname, extname, join, basename } from 'node:path'

export type CorpusCleanerOptions = {
    readonly keepParagraphs?: boolean
    readonly fixHyphenation?: boolean
}

export type CorpusCleanerStats = {
    readonly characterCount: number
    readonly lineCount: number
    readonly paragraphCount: number
}

export type CorpusCleanerResult = {
    readonly cleanedText: string
    readonly before: CorpusCleanerStats
    readonly after: CorpusCleanerStats
}

type CliOptions = CorpusCleanerOptions & {
    readonly path: string
    readonly output: string | undefined
}

if (process.argv[1]?.endsWith('corpus-cleaner.ts') === true) {
    await runCli()
}

export function cleanCorpusText(
    rawText: string,
    options: CorpusCleanerOptions = {},
): CorpusCleanerResult {
    const normalizedLineEndings = rawText.replace(/\r\n?/gu, '\n')
    const textWithoutPageSpacing = trimLines(normalizedLineEndings)
    const textWithoutHyphenation =
        options.fixHyphenation === true
            ? textWithoutPageSpacing.replace(/([\p{L}])-\n([\p{L}])/gu, '$1$2')
            : textWithoutPageSpacing
    const cleanedText = normalizeWhitespace(textWithoutHyphenation, options)

    return {
        after: createCorpusCleanerStats(cleanedText),
        before: createCorpusCleanerStats(rawText),
        cleanedText,
    }
}

export function createDefaultCleanCorpusOutputPath(inputPath: string): string {
    const extension = extname(inputPath)
    const directoryPath = dirname(inputPath)

    if (extension.length === 0) {
        return join(directoryPath, `${basename(inputPath)}.clean`)
    }

    return join(directoryPath, `${basename(inputPath, extension)}.clean${extension}`)
}

export function createCorpusCleanerStats(text: string): CorpusCleanerStats {
    const normalizedText = text.replace(/\r\n?/gu, '\n')
    const paragraphs = normalizedText
        .split(/\n{2,}/u)
        .map((paragraph) => paragraph.trim())
        .filter((paragraph) => paragraph.length > 0)

    return {
        characterCount: Array.from(text).length,
        lineCount: text.length === 0 ? 0 : normalizedText.split('\n').length,
        paragraphCount: paragraphs.length,
    }
}

async function runCli(): Promise<void> {
    const options = parseCliOptions(process.argv.slice(2))
    const outputPath = options.output ?? createDefaultCleanCorpusOutputPath(options.path)
    const rawText = await readFile(options.path, 'utf8')
    const result = cleanCorpusText(rawText, options)

    await writeFile(outputPath, result.cleanedText, 'utf8')

    console.info('Nettoyage corpus terminé.')
    console.info(`Entrée: ${options.path}`)
    console.info(`Sortie: ${outputPath}`)
    console.info('')
    console.info('Avant:')
    printStats(result.before)
    console.info('')
    console.info('Après:')
    printStats(result.after)
}

function trimLines(text: string): string {
    return text
        .split('\n')
        .map((line) => line.trim())
        .join('\n')
}

function normalizeWhitespace(text: string, options: CorpusCleanerOptions): string {
    const paragraphSeparator = '\u0000PARAGRAPH\u0000'
    const withParagraphMarkers = text.replace(/\n{2,}/gu, paragraphSeparator)
    const withoutArtificialLineBreaks = withParagraphMarkers.replace(/\n+/gu, ' ')
    const withOptionalParagraphs =
        options.keepParagraphs === true
            ? withoutArtificialLineBreaks.replaceAll(paragraphSeparator, '\n')
            : withoutArtificialLineBreaks.replaceAll(paragraphSeparator, ' ')

    return withOptionalParagraphs
        .replace(/[ \t\f\v]+/gu, ' ')
        .replace(/ +([,.;:!?])/gu, '$1')
        .replace(/([([{«]) +/gu, '$1')
        .replace(/ +([)\]}»])/gu, '$1')
        .trim()
}

function parseCliOptions(args: readonly string[]): CliOptions {
    let path: string | undefined
    let output: string | undefined
    let keepParagraphs = false
    let fixHyphenation = false

    for (let index = 0; index < args.length; index++) {
        const arg = args[index]

        if (arg === '--path') {
            path = readRequiredArgument(args[index + 1], '--path')
            index++
            continue
        }

        if (arg === '--output') {
            output = readRequiredArgument(args[index + 1], '--output')
            index++
            continue
        }

        if (arg === '--keep-paragraphs') {
            keepParagraphs = true
            continue
        }

        if (arg === '--fix-hyphenation') {
            fixHyphenation = true
            continue
        }

        throw new Error(`Argument inconnu: ${String(arg)}.`)
    }

    if (path === undefined) {
        throw new Error('Commande attendue: npm run corpus:clean -- --path PATH_TO_CORPUS')
    }

    return {
        fixHyphenation,
        keepParagraphs,
        output,
        path,
    }
}

function readRequiredArgument(value: string | undefined, name: string): string {
    if (value === undefined || value.startsWith('--')) {
        throw new Error(`${name} attend une valeur.`)
    }

    return value
}

function printStats(stats: CorpusCleanerStats): void {
    console.info(`  caractères: ${String(stats.characterCount)}`)
    console.info(`  lignes: ${String(stats.lineCount)}`)
    console.info(`  paragraphes: ${String(stats.paragraphCount)}`)
}
