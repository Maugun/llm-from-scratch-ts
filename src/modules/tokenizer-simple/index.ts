export type CharacterTokenizer = {
    /**
     * Table lisible des symboles connus par le tokenizer.
     *
     * Dans ce premier module, un token = un caractere JavaScript.
     * Les LLM modernes utilisent souvent des tokens plus grands, comme des morceaux de mots,
     * mais le principe reste le meme: chaque token recoit un identifiant numerique.
     */
    readonly vocabulary: readonly string[]

    /**
     * Nombre de tokens connus.
     *
     * Plus le vocabulaire est grand, plus les futures tables de probabilites ou d'embeddings
     * devront contenir de lignes. Ici l'impact reste minuscule et uniquement CPU.
     */
    readonly vocabularySize: number

    /**
     * Dictionnaire caractere -> id.
     *
     * L'encodage transforme le texte brut en une sequence discrete d'entiers:
     * "abc" devient par exemple [0, 1, 2].
     */
    readonly charToId: ReadonlyMap<string, number>

    /**
     * Dictionnaire id -> caractere.
     *
     * Le decodage fait l'operation inverse pour revenir a une chaine lisible.
     */
    readonly idToChar: ReadonlyMap<number, string>

    readonly decode: (tokenIds: readonly number[]) => string
    readonly encode: (text: string) => number[]
}

/**
 * Cree un tokenizer caractere a partir d'un texte de reference.
 *
 * Pourquoi un tokenizer existe dans un LLM ?
 * Un modele numerique ne sait pas manipuler directement une chaine de caracteres.
 * Il a besoin d'une entree sous forme de nombres. Le tokenizer est donc la frontiere
 * entre le monde lisible par les humains et le monde manipulable par le modele.
 *
 * Intuition mathematique:
 * un texte est transforme en sequence finie d'entiers. Plus tard, chaque entier pourra
 * indexer une ligne dans une matrice d'embeddings, ou une colonne dans une distribution
 * de probabilites.
 *
 * Memoire / VRAM:
 * ce module ne cree aucun tenseur et n'utilise pas le GPU. La VRAM consommee est donc 0.
 * La RAM utilisee depend seulement du nombre de caracteres uniques et de la longueur
 * des sequences encodees.
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
            throw new Error(`Impossible de decoder l'id inconnu ${String(tokenId)}.`)
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
            throw new Error(`Impossible d'encoder le caractere inconnu "${character}".`)
        }

        tokenIds.push(tokenId)
    }

    return tokenIds
}
