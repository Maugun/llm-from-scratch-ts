import { readFile } from 'node:fs/promises'

export type TextTokenizer = {
    readonly encode: (text: string) => number[]
}

export type TokenDatasetOptions = {
    /**
     * Part du corpus reservee pour la validation.
     *
     * Avec 0.1, on garde environ 10 % des tokens a la fin de la sequence pour mesurer
     * plus tard si un modele apprend autre chose qu'une simple memorisation du train.
     */
    readonly validationRatio?: number
}

export type TokenDataset = {
    /**
     * Texte brut lu depuis le fichier, ou fourni directement a `createTokenDataset`.
     */
    readonly rawText: string

    /**
     * Sequence complete des ids.
     *
     * C'est la forme numerique du corpus: x = [id0, id1, id2, ...].
     */
    readonly tokenIds: readonly number[]

    /**
     * Partie principale du corpus, reservee aux futurs calculs d'apprentissage.
     */
    readonly trainTokenIds: readonly number[]

    /**
     * Petite partie mise de cote pour verifier plus tard le comportement du modele.
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
 * Ce choix est volontairement simple: tout le fichier est charge en RAM CPU. C'est parfait
 * pour un mini corpus pedagogique, mais ce n'est pas une strategie adaptee aux gros datasets.
 */
export async function loadTextFile(filePath: string): Promise<string> {
    return readFile(filePath, 'utf8')
}

/**
 * Transforme un texte brut en dataset de tokens.
 *
 * Role dans un pipeline LLM:
 * - le tokenizer convertit le texte en ids;
 * - le dataset loader organise ces ids pour les prochains modules;
 * - le split train/validation prepare deja la separation entre apprentissage et controle.
 *
 * Memoire / VRAM:
 * ce module utilise seulement de la RAM CPU. Il stocke le texte brut et plusieurs tableaux
 * d'ids. Aucun tenseur n'est cree, donc la VRAM consommee reste 0.
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
            `validationRatio doit etre un nombre fini superieur ou egal a 0 et strictement inferieur a 1. Valeur recue: ${String(
                validationRatio,
            )}.`,
        )
    }
}
