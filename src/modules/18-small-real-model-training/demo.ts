import { existsSync } from 'node:fs'
import { readdir, readFile } from 'node:fs/promises'
import { platform } from 'node:os'
import { join } from 'node:path'

import { createCharacterTokenizer } from '../01-tokenizer-simple/index.js'
import { formatBytes } from '../15-model-sizing-memory-estimator/index.js'
import { loadTfjsNodeGpuBackend } from '../16-tfjs-node-gpu-backend/index.js'
import { createLongCorpusPipeline, loadLongCorpusText } from '../17-long-corpus-pipeline/index.js'
import {
    createSmallLanguageModel,
    disposeSmallLanguageModel,
    evaluateSmallLanguageModel,
    generateSmallLanguageModelText,
    loadSmallLanguageModelCheckpoint,
    normalizeCheckpointVersionName,
    resolveCheckpointVersionPlan,
    saveSmallLanguageModelCheckpoint,
    trainSmallLanguageModel,
    type CheckpointVersionInfo,
    type SmallLanguageModel,
    type SmallLanguageModelGenerationOptions,
    type SmallLanguageModelTrainingProgress,
} from './index.js'

type DemoConfig = {
    readonly contextLength: number
    readonly batchSize: number
    readonly embeddingDimension: number
    readonly feedForwardDimension: number
    readonly layerCount: number
    readonly epochs: number
    readonly learningRate: number
    readonly maxTrainBatchesPerEpoch: number
    readonly maxValidationBatches: number
    readonly maxNewTokens: number
    readonly validationRatio: number
    readonly checkpointPath: string
    readonly checkpointVersion: string | undefined
    readonly batchOrder: 'sequential' | 'shuffled'
    readonly shuffleSeed: number
}

type LocalDemoConfig = Partial<DemoConfig> & {
    readonly corpusPath?: string
}

type CliOptions = {
    readonly continueTrain: boolean
    readonly forceTrain: boolean
    readonly configPath: string
    readonly corpusPath: string | undefined
}

type PreparedPrompt = {
    readonly prompt: string
    readonly wasTruncated: boolean
}

type PendingGeneration = {
    readonly prompt: string
    readonly options: SmallLanguageModelGenerationOptions
}

type PromptTooShortError = Error & {
    readonly missingTokenCount: number
}

const promptTooShortErrorName = 'PromptTooShortError'

const privateDirectoryPath = join(process.cwd(), 'data', 'private')
const fallbackCorpusPath = join(process.cwd(), 'data', 'tiny-corpus.txt')
const defaultConfigPath = join(privateDirectoryPath, 'module-18-config.json')
const defaultCheckpointPath = join(process.cwd(), 'data', 'checkpoints', 'small-real-model')
const cliOptions = parseCliOptions(process.argv.slice(2))
const localConfig = await loadLocalConfig(cliOptions.configPath)

console.info('Module 18 - Entraînement d’un petit modèle réel sur corpus long')
console.info('')
console.info('But du module:')
console.info('  Entraîner un mini Transformer par batches à la demande, puis générer du texte.')
console.info('  Le sampling vient du module 11: greedy, température et top-k.')
console.info('')

const backend = platform() === 'win32' ? undefined : await loadTfjsNodeGpuBackend()

if (backend?.available === true) {
    console.info(`Backend TensorFlow.js: ${backend.backendName}`)
} else if (platform() === 'win32') {
    console.info('Backend TensorFlow.js: @tensorflow/tfjs')
    console.info('Windows natif détecté: le backend GPU CUDA est prévu pour WSL/Linux.')
} else {
    console.info('Backend GPU optionnel non disponible: la démo continue avec @tensorflow/tfjs.')
    console.info(`Raison: ${backend?.errorMessage ?? 'backend non chargé'}`)
}

console.info('')

const corpusPath = await resolveCorpusPath()
const usesPrivateCorpus = corpusPath !== fallbackCorpusPath
const config = createDemoConfig(usesPrivateCorpus, localConfig)
const checkpointPlan = await resolveCheckpointVersionPlan({
    checkpointPath: config.checkpointPath,
    continueTrain: cliOptions.continueTrain,
    forceTrain: cliOptions.forceTrain,
    ...(config.checkpointVersion === undefined
        ? {}
        : { checkpointVersion: config.checkpointVersion }),
})
let checkpointSaveVersion = checkpointPlan.saveVersion
const corpus = await loadLongCorpusText(corpusPath)
const tokenizer = createCharacterTokenizer(corpus.rawText)
const pipeline = createLongCorpusPipeline(corpus.rawText, tokenizer, {
    batchSize: config.batchSize,
    contextLength: config.contextLength,
    validationRatio: config.validationRatio,
})
const modelLoadResult = await createOrLoadModel()
const model = modelLoadResult.model
const defaultPrompt = tokenizer.decode(pipeline.trainTokenIds.slice(0, config.contextLength))

try {
    printIntro()

    if (modelLoadResult.loadedFromCheckpoint && !cliOptions.continueTrain) {
        printLoadedCheckpoint()
        printCheckpointValidation()
        console.info('')
        console.info('Générations depuis checkpoint:')
        printComparisonGenerations(defaultPrompt)
    } else {
        if (modelLoadResult.loadedFromCheckpoint) {
            console.info('Checkpoint existant chargé: poursuite de l’entraînement.')
            console.info(`  dossier chargé: ${modelLoadResult.loadedVersion.directoryPath}`)
            console.info(`  nouvelle version sauvegardée: ${checkpointSaveVersion.directoryPath}`)
            console.info('')
        }

        await trainAndSaveCheckpoint()
    }

    if (process.stdin.isTTY) {
        await startInteractivePrompt()
    } else {
        console.info('')
        console.info(
            'Mode non interactif détecté: lance cette démo dans un terminal pour tester greedy/temp/topk.',
        )
    }
} finally {
    disposeSmallLanguageModel(model)
}

function printIntro(): void {
    console.info('Pipeline:')
    console.info('1. Lire un corpus local')
    console.info('2. Construire un tokenizer caractère')
    console.info('3. Générer les batches à la demande')
    console.info('4. Entraîner ou recharger un mini Transformer')
    console.info('5. Valider sur le split validation')
    console.info('6. Générer avec plusieurs stratégies de sampling')
    console.info('7. Sauvegarder ou réutiliser un checkpoint local')
    console.info('')
    console.info(
        `Config locale: ${existsSync(cliOptions.configPath) ? cliOptions.configPath : 'aucune'}`,
    )
    console.info(`Corpus utilisé: ${corpus.filePath}`)
    console.info(`Taille fichier: ${formatBytes(corpus.stats.byteLength)}`)
    console.info(`Caractères: ${String(corpus.stats.characterCount)}`)
    console.info(`Lignes: ${String(corpus.stats.lineCount)}`)
    console.info(`Vocabulaire: ${String(tokenizer.vocabularySize)} caractères`)
    console.info(`Tokens train: ${String(pipeline.trainTokenCount)}`)
    console.info(`Tokens validation: ${String(pipeline.validationTokenCount)}`)
    console.info('')
    console.info('Configuration modèle:')
    console.info(`  contextLength: ${String(config.contextLength)}`)
    console.info(`  batchSize: ${String(config.batchSize)}`)
    console.info(`  embeddingDimension: ${String(config.embeddingDimension)}`)
    console.info(`  feedForwardDimension: ${String(config.feedForwardDimension)}`)
    console.info(`  layerCount: ${String(config.layerCount)}`)
    console.info(`  epochs: ${String(config.epochs)}`)
    console.info(`  learningRate: ${String(config.learningRate)}`)
    console.info(`  maxTrainBatchesPerEpoch: ${String(config.maxTrainBatchesPerEpoch)}`)
    console.info(`  batchOrder: ${config.batchOrder}`)
    console.info(`  shuffleSeed: ${String(createEffectiveShuffleSeed())}`)
    console.info(`  checkpointPath: ${config.checkpointPath}`)
    console.info(`  checkpointVersion: ${config.checkpointVersion ?? 'aucune'}`)
    console.info(
        `  versions disponibles: ${
            checkpointPlan.availableVersions.length === 0
                ? 'aucune'
                : checkpointPlan.availableVersions.map((version) => version.versionName).join(', ')
        }`,
    )
    console.info(`  version à charger: ${checkpointPlan.loadVersion?.versionName ?? 'aucune'}`)
    console.info(`  version à sauvegarder si entraînement: ${checkpointSaveVersion.versionName}`)
    console.info('')
}

function printLoadedCheckpoint(): void {
    if (!modelLoadResult.loadedFromCheckpoint) {
        return
    }

    console.info('Checkpoint existant chargé:')
    console.info(`  version: ${modelLoadResult.loadedVersion.versionName}`)
    console.info(`  dossier: ${modelLoadResult.loadedVersion.directoryPath}`)
    console.info('  Pour réentraîner depuis zéro dans une nouvelle version:')
    console.info('  npm run demo:18-small-real-model:train')
    console.info('  Pour continuer l’entraînement depuis ce checkpoint:')
    console.info('  npm run demo:18-small-real-model:continue')
}

function printCheckpointValidation(): void {
    const checkpointValidation = evaluateSmallLanguageModel(model, pipeline.validationTokenIds, {
        batchSize: config.batchSize,
        maxBatches: config.maxValidationBatches,
    })

    console.info('')
    console.info(
        `Validation du checkpoint: loss ${checkpointValidation.averageLoss.toFixed(
            4,
        )}, perplexité ${checkpointValidation.perplexity.toFixed(2)}`,
    )
}

async function trainAndSaveCheckpoint(): Promise<void> {
    console.info('Génération avant entraînement:')
    printGeneration(defaultPrompt, {
        maxNewTokens: config.maxNewTokens,
        seed: 123,
        strategy: 'greedy',
    })

    const initialValidation = evaluateSmallLanguageModel(model, pipeline.validationTokenIds, {
        batchSize: config.batchSize,
        maxBatches: config.maxValidationBatches,
    })

    console.info('')
    console.info(
        `Validation avant entraînement: loss ${initialValidation.averageLoss.toFixed(
            4,
        )}, perplexité ${initialValidation.perplexity.toFixed(2)}`,
    )
    console.info('')
    console.info('Entraînement:')
    const trainingStartedAt = Date.now()
    const progressReporter = createTrainingProgressReporter()

    const history = trainSmallLanguageModel(model, pipeline, {
        batchSize: config.batchSize,
        epochs: config.epochs,
        learningRate: config.learningRate,
        maxTrainBatchesPerEpoch: config.maxTrainBatchesPerEpoch,
        maxValidationBatches: config.maxValidationBatches,
        onProgress: progressReporter,
        batchOrder: config.batchOrder,
        shuffleSeed: createEffectiveShuffleSeed(),
    })
    const trainingDurationMs = Date.now() - trainingStartedAt

    finishTrainingProgressLine()

    for (const epoch of history.epochs) {
        console.info(
            `  epoch ${String(epoch.epoch).padStart(2, ' ')} | train loss ${epoch.trainLoss.toFixed(
                4,
            )} | validation loss ${epoch.validationLoss.toFixed(
                4,
            )} | validation perplexité ${epoch.validationPerplexity.toFixed(
                2,
            )} | batches ${String(epoch.trainedBatches)}`,
        )
    }

    console.info('')
    console.info(`Durée d’entraînement: ${formatDuration(trainingDurationMs)}`)
    console.info(
        `Validation après entraînement: loss ${history.finalValidationLoss.toFixed(
            4,
        )}, perplexité ${history.finalValidationPerplexity.toFixed(2)}`,
    )
    console.info('')
    console.info('Générations après entraînement:')
    printComparisonGenerations(defaultPrompt)

    await saveCheckpoint()
}

async function saveCheckpoint(): Promise<void> {
    const checkpoint = await saveSmallLanguageModelCheckpoint(
        model,
        checkpointSaveVersion.directoryPath,
        {
            extra: {
                checkpointVersion: checkpointSaveVersion.versionName,
                corpusPath: corpus.filePath,
                note: 'Checkpoint pédagogique généré par la démo du module 18.',
            },
        },
    )

    console.info('')
    console.info('Checkpoint sauvegardé:')
    console.info(`  version: ${checkpointSaveVersion.versionName}`)
    console.info(`  dossier: ${checkpointSaveVersion.directoryPath}`)
    console.info(`  variables: ${String(checkpoint.variables.length)}`)
}

async function createOrLoadModel(): Promise<
    {
        readonly model: SmallLanguageModel
    } & (
        | {
              readonly loadedFromCheckpoint: true
              readonly loadedVersion: CheckpointVersionInfo
          }
        | {
              readonly loadedFromCheckpoint: false
              readonly loadedVersion: undefined
          }
    )
> {
    if (checkpointPlan.loadVersion !== undefined) {
        try {
            const checkpointModel = await loadSmallLanguageModelCheckpoint(
                checkpointPlan.loadVersion.directoryPath,
            )

            if (isCheckpointModelCompatible(checkpointModel)) {
                return {
                    loadedVersion: checkpointPlan.loadVersion,
                    loadedFromCheckpoint: true,
                    model: checkpointModel,
                }
            }

            disposeSmallLanguageModel(checkpointModel)
            checkpointSaveVersion = checkpointPlan.nextVersion
            console.info(
                'Checkpoint existant ignoré: configuration incompatible avec le corpus courant. La démo sauvegardera dans une nouvelle version.',
            )
            console.info('')
        } catch (error) {
            checkpointSaveVersion = checkpointPlan.nextVersion
            console.info(
                'Checkpoint existant impossible à charger: nouvel entraînement nécessaire. La démo sauvegardera dans une nouvelle version.',
            )
            console.info(error instanceof Error ? `Raison: ${error.message}` : 'Raison inconnue.')
            console.info('')
        }
    }

    return {
        loadedVersion: undefined,
        loadedFromCheckpoint: false,
        model: createSmallLanguageModel({
            contextLength: config.contextLength,
            embeddingDimension: config.embeddingDimension,
            feedForwardDimension: config.feedForwardDimension,
            layerCount: config.layerCount,
            vocabularySize: tokenizer.vocabularySize,
        }),
    }
}

function isCheckpointModelCompatible(checkpointModel: SmallLanguageModel): boolean {
    return (
        checkpointModel.contextLength === config.contextLength &&
        checkpointModel.embeddingDimension === config.embeddingDimension &&
        checkpointModel.feedForwardDimension === config.feedForwardDimension &&
        checkpointModel.layerCount === config.layerCount &&
        checkpointModel.vocabularySize === tokenizer.vocabularySize
    )
}

function printComparisonGenerations(prompt: string): void {
    printGeneration(prompt, {
        maxNewTokens: config.maxNewTokens,
        seed: 123,
        strategy: 'greedy',
    })
    printGeneration(prompt, {
        maxNewTokens: config.maxNewTokens,
        seed: 123,
        strategy: 'temperature',
        temperature: 0.8,
    })
    printGeneration(prompt, {
        maxNewTokens: config.maxNewTokens,
        seed: 123,
        strategy: 'temperature',
        temperature: 1.2,
    })
    printGeneration(prompt, {
        maxNewTokens: config.maxNewTokens,
        seed: 123,
        strategy: 'topK',
        temperature: 1,
        topK: Math.min(5, tokenizer.vocabularySize),
    })
}

async function resolveCorpusPath(): Promise<string> {
    if (cliOptions.corpusPath !== undefined) {
        return cliOptions.corpusPath
    }

    if (localConfig.corpusPath !== undefined) {
        return localConfig.corpusPath
    }

    const preferredPrivatePath = join(privateDirectoryPath, 'long-corpus.txt')

    if (existsSync(preferredPrivatePath)) {
        return preferredPrivatePath
    }

    if (existsSync(privateDirectoryPath)) {
        const entries = await readdir(privateDirectoryPath)
        const firstTextFile = entries.find((entry) => entry.endsWith('.txt'))

        if (firstTextFile !== undefined) {
            return join(privateDirectoryPath, firstTextFile)
        }
    }

    console.info('Aucun corpus privé trouvé: fallback sur data/tiny-corpus.txt.')
    console.info('Pour un vrai essai, place un fichier .txt dans data/private/.')
    console.info('')

    return fallbackCorpusPath
}

function createDemoConfig(
    usesPrivateCorpus: boolean,
    configOverrides: LocalDemoConfig,
): DemoConfig {
    const defaults = createDefaultConfig(usesPrivateCorpus)

    return normalizeConfig({
        ...defaults,
        ...configOverrides,
    })
}

function createDefaultConfig(usesPrivateCorpus: boolean): DemoConfig {
    if (!usesPrivateCorpus) {
        return {
            batchSize: 4,
            batchOrder: 'shuffled',
            checkpointPath: defaultCheckpointPath,
            checkpointVersion: undefined,
            contextLength: 16,
            embeddingDimension: 16,
            epochs: 5,
            feedForwardDimension: 32,
            layerCount: 1,
            learningRate: 0.01,
            maxNewTokens: 60,
            maxTrainBatchesPerEpoch: 20,
            maxValidationBatches: 2,
            shuffleSeed: 18,
            validationRatio: 0.1,
        }
    }

    return {
        batchSize: 16,
        batchOrder: 'shuffled',
        checkpointPath: defaultCheckpointPath,
        checkpointVersion: undefined,
        contextLength: 128,
        embeddingDimension: 64,
        epochs: 1,
        feedForwardDimension: 256,
        layerCount: 1,
        learningRate: 0.001,
        maxNewTokens: 100,
        maxTrainBatchesPerEpoch: 10,
        maxValidationBatches: 2,
        shuffleSeed: 18,
        validationRatio: 0.05,
    }
}

function normalizeConfig(rawConfig: LocalDemoConfig): DemoConfig {
    return {
        batchSize: readPositiveInteger(rawConfig.batchSize, 'batchSize'),
        batchOrder: readBatchOrder(rawConfig.batchOrder),
        checkpointPath: readString(rawConfig.checkpointPath, 'checkpointPath'),
        checkpointVersion: readOptionalCheckpointVersion(rawConfig.checkpointVersion),
        contextLength: readPositiveInteger(rawConfig.contextLength, 'contextLength'),
        embeddingDimension: readPositiveInteger(rawConfig.embeddingDimension, 'embeddingDimension'),
        epochs: readPositiveInteger(rawConfig.epochs, 'epochs'),
        feedForwardDimension: readPositiveInteger(
            rawConfig.feedForwardDimension,
            'feedForwardDimension',
        ),
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
        shuffleSeed: readPositiveInteger(rawConfig.shuffleSeed, 'shuffleSeed'),
        validationRatio: readValidationRatio(rawConfig.validationRatio),
    }
}

function createEffectiveShuffleSeed(): number {
    return config.shuffleSeed + checkpointSaveVersion.versionNumber * 10_000
}

function printGeneration(rawPrompt: string, options: SmallLanguageModelGenerationOptions): void {
    const preparedPrompt = preparePrompt(rawPrompt)
    const result = generateSmallLanguageModelText(model, tokenizer, preparedPrompt.prompt, options)
    const label = formatSamplingLabel(options)

    console.info(`  ${label}:`)

    if (preparedPrompt.wasTruncated) {
        console.info(`    prompt utilisateur: "${rawPrompt}"`)
        console.info(
            `    prompt tronqué aux ${String(config.contextLength)} derniers tokens du message.`,
        )
    }

    console.info(`    prompt utilisé: "${preparedPrompt.prompt}"`)
    console.info(`    texte:         "${result.text}"`)
}

function preparePrompt(rawPrompt: string): PreparedPrompt {
    const promptTokenIds = tokenizer.encode(rawPrompt)

    if (promptTokenIds.length < config.contextLength) {
        throw createPromptTooShortFailure(config.contextLength - promptTokenIds.length)
    }

    if (promptTokenIds.length === config.contextLength) {
        return {
            prompt: rawPrompt,
            wasTruncated: false,
        }
    }

    return {
        prompt: tokenizer.decode(promptTokenIds.slice(-config.contextLength)),
        wasTruncated: true,
    }
}

async function startInteractivePrompt(): Promise<void> {
    console.info('')
    console.info('Mode interactif:')
    console.info('  greedy <prompt court ou long>')
    console.info('  temp <temperature> <prompt court ou long>')
    console.info('  topk <k> <prompt court ou long>')
    console.info('Si le prompt est trop court, la démo garde ta saisie et te demande la suite.')
    console.info('Si le prompt est trop long, elle garde les derniers tokens utiles.')
    console.info('Retour arrière corrige la saisie, Ctrl+U efface la ligne courante.')
    console.info('Appuie sur ENTRÉE pour générer, ou sur ESC pour quitter.')
    console.info('')
    let promptLabel = '> '

    process.stdout.write(promptLabel)

    let currentInput = ''
    let pendingGeneration: PendingGeneration | undefined

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
                pendingGeneration =
                    pendingGeneration === undefined
                        ? handleInteractiveCommand(currentInput)
                        : continuePendingGeneration(pendingGeneration, currentInput)
                currentInput = ''
                promptLabel = pendingGeneration === undefined ? '> ' : 'suite> '
                console.info('')
                process.stdout.write(promptLabel)

                return
            }

            if (input === '\u007F' || input === '\b') {
                currentInput = removeLastCharacter(currentInput)
                redrawInputLine(promptLabel, currentInput)

                return
            }

            if (input === '\u0015') {
                currentInput = ''
                redrawInputLine(promptLabel, currentInput)

                return
            }

            currentInput += input
            process.stdout.write(input)
        }

        process.stdin.on('data', handleInput)
    })
}

function removeLastCharacter(value: string): string {
    return Array.from(value).slice(0, -1).join('')
}

function redrawInputLine(promptLabel: string, currentInput: string): void {
    process.stdout.write(`\r${promptLabel}${currentInput}\x1B[K`)
}

function handleInteractiveCommand(command: string): PendingGeneration | undefined {
    const trimmedCommand = command.trim()

    try {
        if (trimmedCommand.length === 0) {
            return tryPrintInteractiveGeneration(defaultPrompt, {
                maxNewTokens: config.maxNewTokens,
                seed: 123,
                strategy: 'greedy',
            })
        }

        if (trimmedCommand.startsWith('greedy ')) {
            return tryPrintInteractiveGeneration(trimmedCommand.slice('greedy '.length), {
                maxNewTokens: config.maxNewTokens,
                seed: 123,
                strategy: 'greedy',
            })
        }

        if (trimmedCommand.startsWith('temp ')) {
            const parsed = parseNumericCommand(trimmedCommand, 'temp')

            return tryPrintInteractiveGeneration(parsed.prompt, {
                maxNewTokens: config.maxNewTokens,
                seed: 123,
                strategy: 'temperature',
                temperature: parsed.value,
            })
        }

        if (trimmedCommand.startsWith('topk ')) {
            const parsed = parseNumericCommand(trimmedCommand, 'topk')

            return tryPrintInteractiveGeneration(parsed.prompt, {
                maxNewTokens: config.maxNewTokens,
                seed: 123,
                strategy: 'topK',
                topK: parsed.value,
            })
        }

        console.info('Commande inconnue. Utilise: greedy, temp ou topk.')
    } catch (error) {
        console.info(error instanceof Error ? error.message : 'Erreur inconnue.')
    }

    return undefined
}

function continuePendingGeneration(
    pendingGeneration: PendingGeneration,
    additionalText: string,
): PendingGeneration | undefined {
    const nextPrompt = `${pendingGeneration.prompt}${additionalText}`

    return tryPrintInteractiveGeneration(nextPrompt, pendingGeneration.options)
}

function tryPrintInteractiveGeneration(
    prompt: string,
    options: SmallLanguageModelGenerationOptions,
): PendingGeneration | undefined {
    try {
        printGeneration(prompt, options)

        return undefined
    } catch (error) {
        if (isPromptTooShortFailure(error)) {
            console.info(
                `Prompt trop court: ajoute encore environ ${String(
                    error.missingTokenCount,
                )} token(s), puis appuie sur ENTRÉE.`,
            )

            return {
                options,
                prompt,
            }
        }

        throw error
    }
}

function parseCliOptions(args: readonly string[]): CliOptions {
    let continueTrain = false
    let forceTrain = false
    let configPath = defaultConfigPath
    let corpusPath: string | undefined

    for (let index = 0; index < args.length; index++) {
        const arg = args[index]

        if (arg === '--force-train') {
            forceTrain = true
            continue
        }

        if (arg === '--continue-train') {
            continueTrain = true
            continue
        }

        if (arg === '--config') {
            configPath = readRequiredArgument(args[index + 1], '--config')
            index++
            continue
        }

        if (arg === '--corpus') {
            corpusPath = readRequiredArgument(args[index + 1], '--corpus')
            index++
            continue
        }

        throw new Error(`Argument inconnu: ${String(arg)}.`)
    }

    return {
        continueTrain,
        configPath,
        corpusPath,
        forceTrain,
    }
}

async function loadLocalConfig(configPath: string): Promise<LocalDemoConfig> {
    if (!existsSync(configPath)) {
        return {}
    }

    const rawJson = await readFile(configPath, 'utf8')
    const parsedValue = JSON.parse(rawJson) as unknown

    if (!isRecord(parsedValue)) {
        throw new Error('Le fichier de configuration du module 18 doit contenir un objet JSON.')
    }

    return parsedValue
}

function parseNumericCommand(
    command: string,
    commandName: string,
): { readonly value: number; readonly prompt: string } {
    const rest = command.slice(commandName.length).trim()
    const [rawValue, ...promptParts] = rest.split(/\s+/u)

    if (rawValue === undefined || promptParts.length === 0) {
        throw new Error(`Commande attendue: ${commandName} <valeur> <prompt>.`)
    }

    const value = Number(rawValue)

    if (!Number.isFinite(value)) {
        throw new Error(`Valeur invalide: ${rawValue}.`)
    }

    return {
        prompt: promptParts.join(' '),
        value,
    }
}

function createTrainingProgressReporter(): (progress: SmallLanguageModelTrainingProgress) => void {
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

function readRequiredArgument(value: string | undefined, name: string): string {
    if (value === undefined || value.startsWith('--')) {
        throw new Error(`${name} attend une valeur.`)
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

function readValidationRatio(value: unknown): number {
    if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value >= 1) {
        throw new Error('validationRatio doit être un nombre entre 0 inclus et 1 exclu.')
    }

    return value
}

function readString(value: unknown, name: string): string {
    if (typeof value !== 'string' || value.length === 0) {
        throw new Error(`${name} doit être une chaîne non vide.`)
    }

    return value
}

function readBatchOrder(value: unknown): 'sequential' | 'shuffled' {
    if (value === 'sequential' || value === 'shuffled') {
        return value
    }

    throw new Error('batchOrder doit valoir "sequential" ou "shuffled".')
}

function readOptionalCheckpointVersion(value: unknown): string | undefined {
    if (value === undefined) {
        return undefined
    }

    if (typeof value !== 'string' || value.length === 0) {
        throw new Error('checkpointVersion doit être une chaîne non vide.')
    }

    return normalizeCheckpointVersionName(value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
    return typeof value === 'object' && value !== null
}

function createPromptTooShortFailure(missingTokenCount: number): PromptTooShortError {
    const error = new Error('Prompt trop court.') as PromptTooShortError

    error.name = promptTooShortErrorName
    Object.defineProperty(error, 'missingTokenCount', {
        enumerable: true,
        value: missingTokenCount,
    })

    return error
}

function isPromptTooShortFailure(error: unknown): error is PromptTooShortError {
    return (
        error instanceof Error &&
        error.name === promptTooShortErrorName &&
        'missingTokenCount' in error &&
        typeof error.missingTokenCount === 'number'
    )
}

function formatSamplingLabel(options: SmallLanguageModelGenerationOptions): string {
    if (options.strategy === 'greedy') {
        return 'greedy'
    }

    if (options.strategy === 'temperature') {
        return `temperature ${String(options.temperature ?? 1)}`
    }

    return `top-k ${String(options.topK)}`
}
