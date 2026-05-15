import { readFile } from 'node:fs/promises'

export type TextTokenizer = {
    readonly encode: (text: string) => number[]
}

export type TokenDatasetOptions = {
    /**
     * Part du corpus réservée pour la validation.
     *
     * Avec 0.1, on garde environ 10 % des tokens à la fin de la séquence pour mesurer
     * plus tard si un modèle apprend autre chose qu'une simple mémorisation du train.
     */
    readonly validationRatio?: number
}

export type TokenDataset = {
    /**
     * Texte brut lu depuis le fichier, ou fourni directement à `createTokenDataset`.
     */
    readonly rawText: string

    /**
     * Séquence complète des ids.
     *
     * C'est la forme numérique du corpus: x = [id0, id1, id2, ...].
     */
    readonly tokenIds: readonly number[]

    /**
     * Partie principale du corpus, réservée aux futurs calculs d'apprentissage.
     */
    readonly trainTokenIds: readonly number[]

    /**
     * Petite partie mise de côté pour vérifier plus tard le comportement du modèle.
     */
    readonly validationTokenIds: readonly number[]

    readonly totalTokens: number
    readonly trainTokenCount: number
    readonly validationTokenCount: number
}

const defaultValidationRatio = 0.1

/**
 * Lit un fichier texte en UTF-8.
 *
 * Ce choix est volontairement simple: tout le fichier est chargé en RAM CPU. C'est parfait
 * pour un mini corpus pédagogique, mais ce n'est pas une stratégie adaptée aux gros datasets.
 */
export async function loadTextFile(filePath: string): Promise<string> {
    return readFile(filePath, 'utf8')
}

/**
 * Transforme un texte brut en dataset de tokens.
 *
 * Rôle dans un pipeline LLM:
 * - le tokenizer convertit le texte en ids;
 * - le dataset loader organise ces ids pour les prochains modules;
 * - le split train/validation prépare déjà la séparation entre apprentissage et contrôle.
 *
 * Mémoire / VRAM:
 * ce module utilise seulement de la RAM CPU. Il stocke le texte brut et plusieurs tableaux
 * d'ids. Aucun tenseur n'est créé, donc la VRAM consommée reste 0.
 */
export function createTokenDataset(
    rawText: string,
    tokenizer: TextTokenizer,
    options: TokenDatasetOptions = {},
): TokenDataset {
    const validationRatio = options.validationRatio ?? defaultValidationRatio
    validateValidationRatio(validationRatio)

    const tokenIds = tokenizer.encode(rawText)
    const validationTokenCount = Math.floor(tokenIds.length * validationRatio)
    const trainTokenCount = tokenIds.length - validationTokenCount
    const trainTokenIds = tokenIds.slice(0, trainTokenCount)
    const validationTokenIds = tokenIds.slice(trainTokenCount)

    return {
        rawText,
        tokenIds,
        totalTokens: tokenIds.length,
        trainTokenCount,
        trainTokenIds,
        validationTokenCount,
        validationTokenIds,
    }
}

export async function loadTokenDatasetFromFile(
    filePath: string,
    tokenizer: TextTokenizer,
    options: TokenDatasetOptions = {},
): Promise<TokenDataset> {
    const rawText = await loadTextFile(filePath)

    return createTokenDataset(rawText, tokenizer, options)
}

function validateValidationRatio(validationRatio: number): void {
    if (!Number.isFinite(validationRatio) || validationRatio < 0 || validationRatio >= 1) {
        throw new Error(
            `validationRatio doit être un nombre fini supérieur ou égal à 0 et strictement inférieur à 1. Valeur reçue: ${String(
                validationRatio,
            )}.`,
        )
    }
}
