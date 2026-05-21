import { existsSync } from 'node:fs'
import { readdir, readFile } from 'node:fs/promises'
import { createInterface } from 'node:readline/promises'
import { stdin as input, stdout as output } from 'node:process'
import { platform } from 'node:os'
import { join } from 'node:path'

import {
    formatBytes,
    estimateMiniTransformerSize,
} from '../15-model-sizing-memory-estimator/index.js'
import { loadTfjsNodeGpuBackend } from '../16-tfjs-node-gpu-backend/index.js'
import { createLongCorpusPipeline, loadLongCorpusText } from '../17-long-corpus-pipeline/index.js'
import {
    loadFinalTinyLlmCheckpoint,
    createFinalTinyLlm,
    trainBpeTokenizer,
    trainFinalTinyLlm,
    saveFinalTinyLlmCheckpoint,
    disposeFinalTinyLlm,
    generateFinalTinyLlmText,
    evaluateFinalTinyLlm,
    type BpeTokenizer,
    type BpeTokenizerTrainingProgress,
    type FinalTinyLlm,
    type FinalTinyLlmGenerationOptions,
    type FinalTinyLlmTrainingProgress,
} from './index.js'
import {
    resolveCheckpointVersionPlan,
    type CheckpointVersionInfo,
} from '../18-small-real-model-training/index.js'

type DemoMode = 'demo' | 'train' | 'chat' | 'generate'

type FinalTinyLlmDemoConfig = {
    readonly corpusPath: string
    readonly checkpointPath: string
    readonly checkpointVersion: string | undefined
    readonly bpeVocabularySize: number
    readonly bpeMaxTrainingCharacters: number
    readonly contextLength: number
    readonly batchSize: number
    readonly embeddingDimension: number
    readonly feedForwardDimension: number
    readonly layerCount: number
    readonly headCount: number
    readonly epochs: number
    readonly learningRate: number
    readonly maxTrainBatchesPerEpoch: number
    readonly maxValidationBatches: number
    readonly saveBestEpochOnly: boolean
    readonly skipCheckpointWhenNoImprovement: boolean
    readonly validationRatio: number
    readonly batchOrder: 'sequential' | 'shuffled'
    readonly shuffleSeed: number
    readonly maxNewTokens: number
    readonly strategy: 'greedy' | 'temperature' | 'topK'
    readonly temperature: number
    readonly topK: number
    readonly seed: number
    readonly prompt: string
}

type CliOptions = {
    readonly mode: DemoMode
    readonly configPath: string
    readonly forceTrain: boolean
    readonly prompt: string | undefined
}

type ModelLoadResult =
    | {
          readonly loadedFromCheckpoint: true
          readonly loadedVersion: CheckpointVersionInfo
          readonly model: FinalTinyLlm
          readonly tokenizer: BpeTokenizer
      }
    | {
          readonly loadedFromCheckpoint: false
          readonly loadedVersion: undefined
          readonly model: FinalTinyLlm
          readonly tokenizer: BpeTokenizer
      }

const privateDirectoryPath = join(process.cwd(), 'data', 'private')
const fallbackCorpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const defaultConfigPath = join(privateDirectoryPath, 'final-llm-config.json')
const defaultCheckpointPath = join(process.cwd(), 'data', 'checkpoints', 'final-tiny-llm')
const cliOptions = parseCliOptions(process.argv.slice(2))
const localConfig = await loadLocalConfig(cliOptions.configPath)
const corpusPath = await resolveCorpusPath()
const usesPrivateCorpus = corpusPath !== fallbackCorpusPath
const config = createDemoConfig(usesPrivateCorpus, localConfig)
const checkpointPlan = await resolveCheckpointVersionPlan({
    checkpointPath: config.checkpointPath,
    continueTrain: cliOptions.mode === 'train' && !cliOptions.forceTrain,
    forceTrain: cliOptions.forceTrain,
    ...(config.checkpointVersion === undefined
        ? {}
        : { checkpointVersion: config.checkpointVersion }),
})
let checkpointSaveVersion = checkpointPlan.saveVersion
const corpus = await loadLongCorpusText(corpusPath)
let modelLoadResult: ModelLoadResult | undefined

console.info('Module 19 - Tiny LLM final avec BPE, long corpus et chat playground')
console.info('')
console.info('But du module:')
console.info('  Entraîner ou recharger un petit LLM from scratch avec tokenizer BPE,')
console.info('  Transformer multi-head, checkpoints versionnés et génération configurable.')
console.info('')

const backend = platform() === 'win32' ? undefined : await loadTfjsNodeGpuBackend()

if (backend?.available === true) {
    console.info(`Backend TensorFlow.js: ${backend.backendName}`)
} else if (platform() === 'win32') {
    console.info('Backend TensorFlow.js: @tensorflow/tfjs')
    console.info('Windows natif détecté: utilise WSL/Linux pour le backend GPU CUDA.')
} else {
    console.info('Backend GPU optionnel non disponible: fallback @tensorflow/tfjs.')
}

console.info('')

try {
    printConfigSummary()
    modelLoadResult = await createOrLoadModelAndTokenizer()

    if (shouldTrain(modelLoadResult)) {
        await trainAndSave(modelLoadResult)
    } else {
        printLoadedCheckpoint(modelLoadResult)
    }

    if (cliOptions.mode === 'generate') {
        printGeneration(
            modelLoadResult,
            cliOptions.prompt ?? config.prompt,
            createGenerationOptions(),
        )
    } else if (cliOptions.mode === 'chat') {
        await startChat(modelLoadResult)
    } else if (cliOptions.mode === 'demo') {
        printDemoGenerations(modelLoadResult)

        if (process.stdin.isTTY) {
            await startChat(modelLoadResult)
        }
    }
} finally {
    if (modelLoadResult !== undefined) {
        disposeFinalTinyLlm(modelLoadResult.model)
    }
}

function printConfigSummary(): void {
    const parameterEstimate = estimateMiniTransformerSize({
        contextLength: config.contextLength,
        embeddingDimension: config.embeddingDimension,
        feedForwardDimension: config.feedForwardDimension,
        layerCount: config.layerCount,
        vocabularySize: config.bpeVocabularySize,
    })

    console.info(
        `Config locale: ${existsSync(cliOptions.configPath) ? cliOptions.configPath : 'aucune'}`,
    )
    console.info(`Corpus utilisé: ${corpus.filePath}`)
    console.info(`Taille fichier: ${formatBytes(corpus.stats.byteLength)}`)
    console.info(`Caractères: ${String(corpus.stats.characterCount)}`)
    console.info(`Lignes: ${String(corpus.stats.lineCount)}`)
    console.info('')
    console.info('Configuration principale:')
    console.info(`  BPE vocabularySize: ${String(config.bpeVocabularySize)}`)
    console.info(`  contextLength: ${String(config.contextLength)}`)
    console.info(`  batchSize: ${String(config.batchSize)}`)
    console.info(`  embeddingDimension: ${String(config.embeddingDimension)}`)
    console.info(`  headCount: ${String(config.headCount)}`)
    console.info(`  layerCount: ${String(config.layerCount)}`)
    console.info(`  feedForwardDimension: ${String(config.feedForwardDimension)}`)
    console.info(
        `  paramètres estimés: ${parameterEstimate.parameters.total.toLocaleString('fr-FR')}`,
    )
    console.info(
        `  mémoire paramètres estimée: ${formatBytes(parameterEstimate.memory.parameterBytes)}`,
    )
    console.info(`  saveBestEpochOnly: ${String(config.saveBestEpochOnly)}`)
    console.info(
        `  skipCheckpointWhenNoImprovement: ${String(config.skipCheckpointWhenNoImprovement)}`,
    )
    console.info(`  checkpointPath: ${config.checkpointPath}`)
    console.info(`  version à charger: ${checkpointPlan.loadVersion?.versionName ?? 'aucune'}`)
    console.info(`  version à sauvegarder si entraînement: ${checkpointSaveVersion.versionName}`)
    console.info('')
}

async function createOrLoadModelAndTokenizer(): Promise<ModelLoadResult> {
    if (checkpointPlan.loadVersion !== undefined) {
        try {
            const loaded = await loadFinalTinyLlmCheckpoint(
                checkpointPlan.loadVersion.directoryPath,
            )

            if (isCheckpointCompatible(loaded.model)) {
                return {
                    loadedFromCheckpoint: true,
                    loadedVersion: checkpointPlan.loadVersion,
                    model: loaded.model,
                    tokenizer: loaded.tokenizer,
                }
            }

            disposeFinalTinyLlm(loaded.model)
            checkpointSaveVersion = checkpointPlan.nextVersion
            console.info('Checkpoint ignoré: configuration incompatible avec la config courante.')
        } catch (error) {
            checkpointSaveVersion = checkpointPlan.nextVersion
            console.info('Checkpoint impossible à charger: nouvel entraînement nécessaire.')
            console.info(error instanceof Error ? `Raison: ${error.message}` : 'Raison inconnue.')
        }
    }

    console.info('Entraînement du tokenizer BPE:')
    console.info(
        '  Le BPE apprend des morceaux fréquents du corpus pour réduire le nombre de tokens.',
    )
    console.info(
        '  Cette étape peut prendre du temps sur un corpus long: ne quitte pas le terminal tant que la progression avance.',
    )
    const tokenizer = trainBpeTokenizer(corpus.rawText, {
        maxTrainingCharacters: config.bpeMaxTrainingCharacters,
        onProgress: createBpeProgressReporter(),
        vocabularySize: config.bpeVocabularySize,
    })
    finishTrainingProgressLine()
    const model = createFinalTinyLlm({
        contextLength: config.contextLength,
        embeddingDimension: config.embeddingDimension,
        feedForwardDimension: config.feedForwardDimension,
        headCount: config.headCount,
        layerCount: config.layerCount,
        vocabularySize: tokenizer.vocabularySize,
    })

    return {
        loadedFromCheckpoint: false,
        loadedVersion: undefined,
        model,
        tokenizer,
    }
}

function shouldTrain(result: ModelLoadResult): boolean {
    return !result.loadedFromCheckpoint || cliOptions.mode === 'train' || cliOptions.forceTrain
}

async function trainAndSave(result: ModelLoadResult): Promise<void> {
    const pipeline = createLongCorpusPipeline(corpus.rawText, result.tokenizer, {
        batchSize: config.batchSize,
        contextLength: config.contextLength,
        validationRatio: config.validationRatio,
    })

    console.info('')
    console.info('Pipeline d’entraînement:')
    console.info(`  vocabulaire BPE effectif: ${String(result.tokenizer.vocabularySize)} tokens`)
    console.info(`  tokens totaux: ${String(pipeline.totalTokens)}`)
    console.info(`  exemples train estimés: ${String(pipeline.trainExampleCount)}`)
    console.info(`  batches train disponibles: ${String(pipeline.trainBatchCount)}`)
    console.info('')

    const initialValidation = evaluateFinalTinyLlm(result.model, pipeline.validationTokenIds, {
        batchSize: config.batchSize,
        maxBatches: config.maxValidationBatches,
    })

    console.info(
        `Validation avant entraînement: loss ${initialValidation.averageLoss.toFixed(
            4,
        )}, perplexité ${initialValidation.perplexity.toFixed(2)}`,
    )
    console.info('Entraînement:')
    const startedAt = Date.now()
    const history = trainFinalTinyLlm(result.model, pipeline, {
        batchOrder: config.batchOrder,
        batchSize: config.batchSize,
        epochs: config.epochs,
        learningRate: config.learningRate,
        maxTrainBatchesPerEpoch: config.maxTrainBatchesPerEpoch,
        maxValidationBatches: config.maxValidationBatches,
        onProgress: createTrainingProgressReporter(),
        saveBestEpochOnly: config.saveBestEpochOnly,
        shuffleSeed: config.shuffleSeed + checkpointSaveVersion.versionNumber * 10_000,
    })
    const durationMs = Date.now() - startedAt

    finishTrainingProgressLine()

    for (const epoch of history.epochs) {
        console.info(
            `  epoch ${String(epoch.epoch).padStart(2, ' ')} | train loss ${epoch.trainLoss.toFixed(
                4,
            )} | validation loss ${epoch.validationLoss.toFixed(
                4,
            )} | validation perplexité ${epoch.validationPerplexity.toFixed(2)}`,
        )
    }

    console.info(`Durée d’entraînement: ${formatDuration(durationMs)}`)
    console.info(
        `Validation après entraînement: loss ${history.finalValidationLoss.toFixed(
            4,
        )}, perplexité ${history.finalValidationPerplexity.toFixed(2)}`,
    )
    console.info(
        `Meilleur point du run: ${
            history.bestEpoch === 0 ? 'avant entraînement' : `epoch ${String(history.bestEpoch)}`
        } | validation loss ${history.bestValidationLoss.toFixed(
            4,
        )} | perplexité ${history.bestValidationPerplexity.toFixed(2)}`,
    )

    if (history.restoredBestEpochWeights) {
        console.info('Checkpoint: les poids sauvegardés correspondent au meilleur point du run.')
    }

    if (
        config.skipCheckpointWhenNoImprovement &&
        history.bestValidationLoss >= history.initialValidationLoss
    ) {
        console.info('')
        console.info('Checkpoint non sauvegardé:')
        console.info(
            `  aucune epoch n’a amélioré la validation initiale (${history.initialValidationLoss.toFixed(
                4,
            )}).`,
        )
        console.info(
            '  Les poids en mémoire ont été restaurés au meilleur point du run, mais aucune nouvelle version inutile n’a été écrite.',
        )

        return
    }

    const metadata = await saveFinalTinyLlmCheckpoint(
        result.model,
        result.tokenizer,
        checkpointSaveVersion.directoryPath,
        {
            extra: {
                config,
                corpusPath: corpus.filePath,
                metrics: history,
                note: 'Checkpoint pédagogique du module 19.',
            },
        },
    )

    console.info('')
    console.info('Checkpoint sauvegardé:')
    console.info(`  version: ${checkpointSaveVersion.versionName}`)
    console.info(`  dossier: ${checkpointSaveVersion.directoryPath}`)
    console.info(`  variables: ${String(metadata.variables.length)}`)
}

function printLoadedCheckpoint(result: ModelLoadResult): void {
    if (!result.loadedFromCheckpoint) {
        return
    }

    console.info('Checkpoint existant chargé:')
    console.info(`  version: ${result.loadedVersion.versionName}`)
    console.info(`  dossier: ${result.loadedVersion.directoryPath}`)
}

function printDemoGenerations(result: ModelLoadResult): void {
    console.info('')
    console.info('Générations comparées:')
    const options = createGenerationOptions()

    printGeneration(result, config.prompt, { ...options, strategy: 'greedy' })
    printGeneration(result, config.prompt, {
        ...options,
        strategy: 'temperature',
        temperature: 0.8,
    })
    printGeneration(result, config.prompt, {
        ...options,
        strategy: 'temperature',
        temperature: 1.2,
    })
    printGeneration(result, config.prompt, {
        ...options,
        strategy: 'topK',
        topK: Math.min(config.topK, result.tokenizer.vocabularySize),
    })
}

function printGeneration(
    result: ModelLoadResult,
    rawPrompt: string,
    options: FinalTinyLlmGenerationOptions,
): void {
    const prompt = preparePrompt(result.tokenizer, rawPrompt)
    const generation = generateFinalTinyLlmText(result.model, result.tokenizer, prompt, options)

    console.info(`  ${formatSamplingLabel(options)}:`)
    console.info(`    prompt utilisé: "${prompt}"`)
    console.info(`    texte: "${generation.text}"`)
}

async function startChat(result: ModelLoadResult): Promise<void> {
    console.info('')
    console.info('Chat playground:')
    console.info('  Ce mode n’est pas un vrai assistant instruction-tuned.')
    console.info('  Il formate seulement ton message en prompt puis génère la suite.')
    console.info('  Tape "exit" pour quitter.')
    console.info('')

    const readline = createInterface({ input, output })
    let shouldContinue = true

    try {
        while (shouldContinue) {
            const message = await readline.question('Utilisateur> ')

            if (message.trim().toLowerCase() === 'exit') {
                shouldContinue = false
                continue
            }

            try {
                const prompt = preparePrompt(result.tokenizer, message)
                const generation = generateFinalTinyLlmText(
                    result.model,
                    result.tokenizer,
                    prompt,
                    createGenerationOptions(),
                )

                console.info(`Assistant> ${generation.generatedText}`)
            } catch (error) {
                console.info(
                    error instanceof Error
                        ? `Message non générable: ${error.message}`
                        : 'Message non générable: erreur inconnue.',
                )
                console.info(
                    'Indice: avec un tokenizer entraîné localement, seuls les caractères présents dans le corpus sont connus.',
                )
            }
        }
    } finally {
        readline.close()
    }
}

function preparePrompt(tokenizer: BpeTokenizer, rawPrompt: string): string {
    const tokenIds = tokenizer.encode(rawPrompt)

    if (tokenIds.length < config.contextLength) {
        const paddingSource = tokenizer.decode(
            tokenizer.encode(corpus.rawText).slice(0, config.contextLength),
        )
        const paddedPrompt = `${paddingSource} ${rawPrompt}`
        const paddedTokenIds = tokenizer.encode(paddedPrompt)

        return tokenizer.decode(paddedTokenIds.slice(-config.contextLength))
    }

    if (tokenIds.length > config.contextLength) {
        return tokenizer.decode(tokenIds.slice(-config.contextLength))
    }

    return rawPrompt
}

function createGenerationOptions(): FinalTinyLlmGenerationOptions {
    return {
        maxNewTokens: config.maxNewTokens,
        seed: config.seed,
        strategy: config.strategy,
        temperature: config.temperature,
        topK: config.topK,
    }
}

async function resolveCorpusPath(): Promise<string> {
    if (typeof localConfig.corpusPath === 'string') {
        return localConfig.corpusPath
    }

    const preferredPrivatePath = join(privateDirectoryPath, 'long-corpus.clean.txt')

    if (existsSync(preferredPrivatePath)) {
        return preferredPrivatePath
    }

    if (existsSync(privateDirectoryPath)) {
        const entries = await readdir(privateDirectoryPath)
        const firstTextFile = entries.find(
            (entry) => entry.endsWith('.clean.txt') || entry.endsWith('.txt'),
        )

        if (firstTextFile !== undefined) {
            return join(privateDirectoryPath, firstTextFile)
        }
    }

    console.info('Aucun corpus privé trouvé: fallback sur data/tiny-corpus.txt.')
    console.info('Pour un vrai essai, nettoie un corpus local avec npm run corpus:clean.')
    console.info('')

    return fallbackCorpusPath
}

function createDemoConfig(
    usesPrivateCorpus: boolean,
    configOverrides: Record<string, unknown>,
): FinalTinyLlmDemoConfig {
    const defaults = usesPrivateCorpus ? createPrivateCorpusDefaults() : createFallbackDefaults()

    return normalizeConfig({ ...defaults, ...configOverrides })
}

function createFallbackDefaults(): FinalTinyLlmDemoConfig {
    return {
        batchOrder: 'shuffled',
        batchSize: 4,
        bpeMaxTrainingCharacters: 20_000,
        bpeVocabularySize: 80,
        checkpointPath: defaultCheckpointPath,
        checkpointVersion: undefined,
        contextLength: 12,
        corpusPath,
        embeddingDimension: 32,
        epochs: 2,
        feedForwardDimension: 64,
        headCount: 4,
        layerCount: 1,
        learningRate: 0.003,
        maxNewTokens: 80,
        maxTrainBatchesPerEpoch: 10,
        maxValidationBatches: 2,
        prompt: 'bonjour le',
        saveBestEpochOnly: false,
        skipCheckpointWhenNoImprovement: false,
        seed: 19,
        shuffleSeed: 19,
        strategy: 'topK',
        temperature: 0.9,
        topK: 5,
        validationRatio: 0.4,
    }
}

function createPrivateCorpusDefaults(): FinalTinyLlmDemoConfig {
    return {
        batchOrder: 'shuffled',
        batchSize: 16,
        bpeMaxTrainingCharacters: 300_000,
        bpeVocabularySize: 1_000,
        checkpointPath: defaultCheckpointPath,
        checkpointVersion: undefined,
        contextLength: 128,
        corpusPath,
        embeddingDimension: 128,
        epochs: 1,
        feedForwardDimension: 512,
        headCount: 4,
        layerCount: 3,
        learningRate: 0.0005,
        maxNewTokens: 200,
        maxTrainBatchesPerEpoch: 100,
        maxValidationBatches: 10,
        prompt: 'Utilisateur: Bonjour\nAssistant:',
        saveBestEpochOnly: false,
        skipCheckpointWhenNoImprovement: false,
        seed: 19,
        shuffleSeed: 19,
        strategy: 'topK',
        temperature: 0.9,
        topK: 20,
        validationRatio: 0.05,
    }
}

function normalizeConfig(rawConfig: Record<string, unknown>): FinalTinyLlmDemoConfig {
    return {
        batchOrder: readBatchOrder(rawConfig.batchOrder),
        batchSize: readPositiveInteger(rawConfig.batchSize, 'batchSize'),
        bpeMaxTrainingCharacters: readPositiveInteger(
            rawConfig.bpeMaxTrainingCharacters,
            'bpeMaxTrainingCharacters',
        ),
        bpeVocabularySize: readPositiveInteger(rawConfig.bpeVocabularySize, 'bpeVocabularySize'),
        checkpointPath: readString(rawConfig.checkpointPath, 'checkpointPath'),
        checkpointVersion: readOptionalString(rawConfig.checkpointVersion),
        contextLength: readPositiveInteger(rawConfig.contextLength, 'contextLength'),
        corpusPath: readString(rawConfig.corpusPath, 'corpusPath'),
        embeddingDimension: readPositiveInteger(rawConfig.embeddingDimension, 'embeddingDimension'),
        epochs: readPositiveInteger(rawConfig.epochs, 'epochs'),
        feedForwardDimension: readPositiveInteger(
            rawConfig.feedForwardDimension,
            'feedForwardDimension',
        ),
        headCount: readPositiveInteger(rawConfig.headCount, 'headCount'),
        layerCount: readPositiveInteger(rawConfig.layerCount, 'layerCount'),
        learningRate: readPositiveNumber(rawConfig.learningRate, 'learningRate'),
        maxNewTokens: readPositiveInteger(rawConfig.maxNewTokens, 'maxNewTokens'),
        maxTrainBatchesPerEpoch: readPositiveInteger(
            rawConfig.maxTrainBatchesPerEpoch,
            'maxTrainBatchesPerEpoch',
        ),
        maxValidationBatches: readPositiveInteger(
            rawConfig.maxValidationBatches,
            'maxValidationBatches',
        ),
        prompt: readString(rawConfig.prompt, 'prompt'),
        saveBestEpochOnly: readOptionalBoolean(rawConfig.saveBestEpochOnly, 'saveBestEpochOnly'),
        seed: readPositiveInteger(rawConfig.seed, 'seed'),
        shuffleSeed: readPositiveInteger(rawConfig.shuffleSeed, 'shuffleSeed'),
        skipCheckpointWhenNoImprovement: readOptionalBoolean(
            rawConfig.skipCheckpointWhenNoImprovement,
            'skipCheckpointWhenNoImprovement',
        ),
        strategy: readStrategy(rawConfig.strategy),
        temperature: readPositiveNumber(rawConfig.temperature, 'temperature'),
        topK: readPositiveInteger(rawConfig.topK, 'topK'),
        validationRatio: readValidationRatio(rawConfig.validationRatio),
    }
}

function isCheckpointCompatible(model: FinalTinyLlm): boolean {
    return (
        model.contextLength === config.contextLength &&
        model.embeddingDimension === config.embeddingDimension &&
        model.feedForwardDimension === config.feedForwardDimension &&
        model.headCount === config.headCount &&
        model.layerCount === config.layerCount
    )
}

function parseCliOptions(args: readonly string[]): CliOptions {
    let mode: DemoMode = 'demo'
    let configPath = defaultConfigPath
    let forceTrain = false
    let prompt: string | undefined

    for (let index = 0; index < args.length; index++) {
        const arg = args[index]

        if (arg === '--mode') {
            mode = readMode(readRequiredArgument(args[index + 1], '--mode'))
            index++
            continue
        }

        if (arg === '--config') {
            configPath = readRequiredArgument(args[index + 1], '--config')
            index++
            continue
        }

        if (arg === '--force-train') {
            forceTrain = true
            continue
        }

        if (arg === '--prompt') {
            prompt = readRequiredArgument(args[index + 1], '--prompt')
            index++
            continue
        }

        throw new Error(`Argument inconnu: ${String(arg)}.`)
    }

    return { configPath, forceTrain, mode, prompt }
}

async function loadLocalConfig(configPath: string): Promise<Record<string, unknown>> {
    if (!existsSync(configPath)) {
        return {}
    }

    const rawJson = await readFile(configPath, 'utf8')
    const parsedValue = JSON.parse(rawJson) as unknown

    if (!isRecord(parsedValue)) {
        throw new Error('La configuration du module 19 doit être un objet JSON.')
    }

    return parsedValue
}

function createTrainingProgressReporter(): (progress: FinalTinyLlmTrainingProgress) => void {
    let lastPrintedAt = 0
    let lastPrintedPercent = -1

    return (progress) => {
        const percent = Math.floor(progress.progressRatio * 100)
        const now = Date.now()
        const shouldPrint =
            percent === 100 ||
            percent >= lastPrintedPercent + 10 ||
            now - lastPrintedAt >= 1_000 ||
            progress.trainedBatches === 1

        if (!shouldPrint) {
            return
        }

        lastPrintedAt = now
        lastPrintedPercent = percent
        const message = `  epoch ${String(progress.epoch).padStart(2, ' ')}/${String(
            progress.epochs,
        )} | ${String(percent).padStart(3, ' ')}% | batch ${String(
            progress.trainedBatches,
        )}/${String(progress.totalBatchesInEpoch)} | loss batch ${progress.latestBatchLoss.toFixed(
            4,
        )} | ${formatDuration(progress.elapsedMs)}`

        if (process.stdout.isTTY) {
            process.stdout.write(`\r${message}\x1B[K`)
            return
        }

        console.info(message)
    }
}

function createBpeProgressReporter(): (progress: BpeTokenizerTrainingProgress) => void {
    let lastPrintedAt = 0
    let lastPrintedPercent = -1

    return (progress) => {
        const percent = Math.floor(progress.progressRatio * 100)
        const now = Date.now()
        const shouldPrint =
            percent === 100 ||
            percent >= lastPrintedPercent + 10 ||
            now - lastPrintedAt >= 1_000 ||
            progress.mergeCount === 1

        if (!shouldPrint) {
            return
        }

        lastPrintedAt = now
        lastPrintedPercent = percent
        const latestMerge =
            progress.latestMerge === undefined
                ? 'aucun merge'
                : `"${progress.latestMerge.left}" + "${progress.latestMerge.right}" -> "${progress.latestMerge.merged}"`
        const message = `  BPE | ${String(percent).padStart(3, ' ')}% | vocab ${String(
            progress.vocabularySize,
        )}/${String(progress.targetVocabularySize)} | merges ${String(
            progress.mergeCount,
        )} | ${latestMerge} | ${formatDuration(progress.elapsedMs)}`

        if (process.stdout.isTTY) {
            process.stdout.write(`\r${message}\x1B[K`)
            return
        }

        console.info(message)
    }
}

function finishTrainingProgressLine(): void {
    if (process.stdout.isTTY) {
        console.info('')
    }
}

function formatDuration(durationMs: number): string {
    const totalSeconds = Math.max(0, Math.round(durationMs / 1_000))
    const minutes = Math.floor(totalSeconds / 60)
    const seconds = totalSeconds % 60

    if (minutes === 0) {
        return `${String(seconds)}s`
    }

    return `${String(minutes)}min ${String(seconds).padStart(2, '0')}s`
}

function formatSamplingLabel(options: FinalTinyLlmGenerationOptions): string {
    if (options.strategy === 'greedy') {
        return 'greedy'
    }

    if (options.strategy === 'temperature') {
        return `temperature ${String(options.temperature ?? 1)}`
    }

    return `top-k ${String(options.topK)}`
}

function readRequiredArgument(value: string | undefined, name: string): string {
    if (value === undefined || value.startsWith('--')) {
        throw new Error(`${name} attend une valeur.`)
    }

    return value
}

function readMode(value: string): DemoMode {
    if (value === 'demo' || value === 'train' || value === 'chat' || value === 'generate') {
        return value
    }

    throw new Error('--mode doit valoir demo, train, chat ou generate.')
}

function readStrategy(value: unknown): 'greedy' | 'temperature' | 'topK' {
    if (value === 'greedy' || value === 'temperature' || value === 'topK') {
        return value
    }

    throw new Error('strategy doit valoir greedy, temperature ou topK.')
}

function readBatchOrder(value: unknown): 'sequential' | 'shuffled' {
    if (value === 'sequential' || value === 'shuffled') {
        return value
    }

    throw new Error('batchOrder doit valoir sequential ou shuffled.')
}

function readString(value: unknown, name: string): string {
    if (typeof value !== 'string' || value.length === 0) {
        throw new Error(`${name} doit être une chaîne non vide.`)
    }

    return value
}

function readOptionalString(value: unknown): string | undefined {
    if (value === undefined) {
        return undefined
    }

    if (typeof value !== 'string' || value.length === 0) {
        throw new Error('checkpointVersion doit être une chaîne non vide.')
    }

    return value
}

function readPositiveInteger(value: unknown, name: string): number {
    if (typeof value !== 'number' || !Number.isInteger(value) || value <= 0) {
        throw new Error(`${name} doit être un entier strictement positif.`)
    }

    return value
}

function readPositiveNumber(value: unknown, name: string): number {
    if (typeof value !== 'number' || !Number.isFinite(value) || value <= 0) {
        throw new Error(`${name} doit être un nombre fini strictement positif.`)
    }

    return value
}

function readOptionalBoolean(value: unknown, name: string): boolean {
    if (value === undefined) {
        return false
    }

    if (typeof value !== 'boolean') {
        throw new Error(`${name} doit être un booléen.`)
    }

    return value
}

function readValidationRatio(value: unknown): number {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value >= 1) {
        throw new Error('validationRatio doit être un nombre entre 0 inclus et 1 exclu.')
    }

    return value
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null
}
