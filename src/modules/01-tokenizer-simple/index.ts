export type CharacterTokenizer = {
    /**
     * Table lisible des symboles connus par le tokenizer.
     *
     * Dans ce premier module, un token = un caractère JavaScript.
     * Les LLM modernes utilisent souvent des tokens plus grands, comme des morceaux de mots,
     * mais le principe reste le même: chaque token reçoit un identifiant numérique.
     */
    readonly vocabulary: readonly string[]

    /**
     * Nombre de tokens connus.
     *
     * Plus le vocabulaire est grand, plus les futures tables de probabilités ou d'embeddings
     * devront contenir de lignes. Ici l'impact reste minuscule et uniquement CPU.
     */
    readonly vocabularySize: number

    /**
     * Dictionnaire caractère -> id.
     *
     * L'encodage transforme le texte brut en une séquence discrète d'entiers:
     * "abc" devient par exemple [0, 1, 2].
     */
    readonly charToId: ReadonlyMap<string, number>

    /**
     * Dictionnaire id -> caractère.
     *
     * Le décodage fait l'opération inverse pour revenir à une chaîne lisible.
     */
    readonly idToChar: ReadonlyMap<number, string>

    readonly decode: (tokenIds: readonly number[]) => string
    readonly encode: (text: string) => number[]
}

/**
 * Crée un tokenizer caractère à partir d'un texte de référence.
 *
 * Pourquoi un tokenizer existe dans un LLM ?
 * Un modèle numérique ne sait pas manipuler directement une chaîne de caractères.
 * Il a besoin d'une entrée sous forme de nombres. Le tokenizer est donc la frontière
 * entre le monde lisible par les humains et le monde manipulable par le modèle.
 *
 * Intuition mathématique:
 * un texte est transformé en séquence finie d'entiers. Plus tard, chaque entier pourra
 * indexer une ligne dans une matrice d'embeddings, ou une colonne dans une distribution
 * de probabilités.
 *
 * Mémoire / VRAM:
 * ce module ne crée aucun tenseur et n'utilise pas le GPU. La VRAM consommée est donc 0.
 * La RAM utilisée dépend seulement du nombre de caractères uniques et de la longueur
 * des séquences encodées.
 */
export function createCharacterTokenizer(trainingText: string): CharacterTokenizer {
    const vocabulary = Array.from(new Set(Array.from(trainingText))).sort()
    const charToId = new Map<string, number>()
    const idToChar = new Map<number, string>()

    for (const [id, character] of vocabulary.entries()) {
        charToId.set(character, id)
        idToChar.set(id, character)
    }

    return {
        charToId,
        decode: (tokenIds) => decodeCharacters(tokenIds, idToChar),
        encode: (text) => encodeCharacters(text, charToId),
        idToChar,
        vocabulary,
        vocabularySize: vocabulary.length,
    }
}

function decodeCharacters(
    tokenIds: readonly number[],
    idToChar: ReadonlyMap<number, string>,
): string {
    const characters: string[] = []

    for (const tokenId of tokenIds) {
        const character = idToChar.get(tokenId)

        if (character === undefined) {
            throw new Error(`Impossible de décoder l'id inconnu ${String(tokenId)}.`)
        }

        characters.push(character)
    }

    return characters.join('')
}

function encodeCharacters(text: string, charToId: ReadonlyMap<string, number>): number[] {
    const tokenIds: number[] = []

    for (const character of text) {
        const tokenId = charToId.get(character)

        if (tokenId === undefined) {
            throw new Error(`Impossible d'encoder le caractère inconnu "${character}".`)
        }

        tokenIds.push(tokenId)
    }

    return tokenIds
}
