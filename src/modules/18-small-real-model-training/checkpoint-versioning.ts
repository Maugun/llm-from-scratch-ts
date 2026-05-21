import { access, readdir } from 'node:fs/promises'
import { join } from 'node:path'

export type CheckpointVersionInfo = {
    readonly versionName: string
    readonly versionNumber: number
    readonly directoryPath: string
}

export type CheckpointVersionPlanOptions = {
    readonly checkpointPath: string
    readonly checkpointVersion?: string
    readonly continueTrain?: boolean
    readonly forceTrain?: boolean
}

export type CheckpointVersionPlan = {
    readonly checkpointPath: string
    readonly requestedVersion: CheckpointVersionInfo | undefined
    readonly latestVersion: CheckpointVersionInfo | undefined
    readonly loadVersion: CheckpointVersionInfo | undefined
    readonly saveVersion: CheckpointVersionInfo
    readonly nextVersion: CheckpointVersionInfo
    readonly availableVersions: readonly CheckpointVersionInfo[]
}

const checkpointMetadataFileName = 'metadata.json'
const checkpointVersionPattern = /^v([1-9]\d*)$/u

export async function listCheckpointVersions(
    checkpointPath: string,
): Promise<readonly CheckpointVersionInfo[]> {
    let entries: readonly { readonly name: string; isDirectory(): boolean }[]

    try {
        entries = await readdir(checkpointPath, { withFileTypes: true })
    } catch (error) {
        if (isNodeError(error) && error.code === 'ENOENT') {
            return []
        }

        throw error
    }

    const versions: CheckpointVersionInfo[] = []

    for (const entry of entries) {
        if (!entry.isDirectory()) {
            continue
        }

        const parsedVersion = parseCheckpointVersionName(entry.name)

        if (parsedVersion === undefined) {
            continue
        }

        const directoryPath = join(checkpointPath, entry.name)

        if (!(await hasCheckpointMetadata(directoryPath))) {
            continue
        }

        versions.push({
            directoryPath,
            versionName: entry.name,
            versionNumber: parsedVersion,
        })
    }

    return versions.sort((left, right) => left.versionNumber - right.versionNumber)
}

export async function resolveCheckpointVersionPlan(
    options: CheckpointVersionPlanOptions,
): Promise<CheckpointVersionPlan> {
    const availableVersions = await listCheckpointVersions(options.checkpointPath)
    const latestVersion = availableVersions.at(-1)
    const requestedVersion =
        options.checkpointVersion === undefined
            ? undefined
            : createVersionInfo(
                  options.checkpointPath,
                  normalizeCheckpointVersionName(options.checkpointVersion),
              )
    const matchingRequestedVersion =
        requestedVersion === undefined
            ? undefined
            : availableVersions.find(
                  (version) => version.versionNumber === requestedVersion.versionNumber,
              )
    const loadVersion =
        options.forceTrain === true
            ? undefined
            : (matchingRequestedVersion ??
              (requestedVersion === undefined ? latestVersion : undefined))
    const nextVersion = createVersionInfo(
        options.checkpointPath,
        formatCheckpointVersionName((latestVersion?.versionNumber ?? 0) + 1),
    )
    const saveVersion = resolveSaveVersion({
        checkpointPath: options.checkpointPath,
        continueTrain: options.continueTrain === true,
        forceTrain: options.forceTrain === true,
        latestVersion,
        nextVersion,
        requestedVersion,
    })

    return {
        availableVersions,
        checkpointPath: options.checkpointPath,
        latestVersion,
        loadVersion,
        nextVersion,
        requestedVersion,
        saveVersion,
    }
}

export function normalizeCheckpointVersionName(versionName: string): string {
    const numericVersionPattern = /^([1-9]\d*)$/u

    if (checkpointVersionPattern.test(versionName)) {
        return versionName
    }

    if (numericVersionPattern.test(versionName)) {
        return `v${versionName}`
    }

    throw new Error('checkpointVersion doit être au format "v1", "v2", "v3", etc.')
}

function resolveSaveVersion(options: {
    readonly checkpointPath: string
    readonly continueTrain: boolean
    readonly forceTrain: boolean
    readonly latestVersion: CheckpointVersionInfo | undefined
    readonly nextVersion: CheckpointVersionInfo
    readonly requestedVersion: CheckpointVersionInfo | undefined
}): CheckpointVersionInfo {
    if (options.continueTrain || options.forceTrain) {
        return options.nextVersion
    }

    return (
        options.requestedVersion ??
        options.latestVersion ??
        createVersionInfo(options.checkpointPath, 'v1')
    )
}

function parseCheckpointVersionName(versionName: string): number | undefined {
    const match = checkpointVersionPattern.exec(versionName)

    if (match?.[1] === undefined) {
        return undefined
    }

    return Number(match[1])
}

function formatCheckpointVersionName(versionNumber: number): string {
    return `v${String(versionNumber)}`
}

function createVersionInfo(checkpointPath: string, versionName: string): CheckpointVersionInfo {
    const versionNumber = parseCheckpointVersionName(versionName)

    if (versionNumber === undefined) {
        throw new Error('checkpointVersion doit être au format "v1", "v2", "v3", etc.')
    }

    return {
        directoryPath: join(checkpointPath, versionName),
        versionName,
        versionNumber,
    }
}

async function hasCheckpointMetadata(directoryPath: string): Promise<boolean> {
    try {
        await access(join(directoryPath, checkpointMetadataFileName))

        return true
    } catch (error) {
        if (isNodeError(error) && error.code === 'ENOENT') {
            return false
        }

        throw error
    }
}

function isNodeError(error: unknown): error is NodeJS.ErrnoException {
    return error instanceof Error && 'code' in error
}
