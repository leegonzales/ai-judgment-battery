/**
 * Import dilemmas and model responses from the AI Judgment Battery data.
 *
 * Usage: npx tsx scripts/import_battery.ts [--dry-run] [--clear]
 *        [--dilemmas path] [--results path1 path2 ...]
 *
 * Flags:
 *   --dry-run   Show what would be imported without writing to DB
 *   --clear     Wipe existing data before importing
 *
 * Defaults to ../dilemmas/all_dilemmas.json and auto-discovers the latest
 * result file per target model (claude-opus, gpt-5.1, gemini-3-pro).
 */

import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import * as schema from "../src/db/schema";
import fs from "fs";
import path from "path";
import { v4 as uuid } from "uuid";

const DB_PATH = path.join(__dirname, "..", "data", "ethics-arena.db");
const BATTERY_ROOT = path.join(__dirname, "..", "..");

const TARGET_MODEL_KEYS = ["claude-opus", "gpt-5.1", "gemini-3-pro"];

interface DilemmaFile {
    version: string;
    description: string;
    categories: Record<string, string>;
    dilemmas: Array<{
        id: string;
        category: string;
        title: string;
        scenario: string;
        question: string;
    }>;
}

interface ResultFile {
    run_id: string;
    model_key: string;
    model: string;
    responses: Array<{
        dilemma_id: string;
        response: string;
        [key: string]: unknown;
    }>;
}

function getFlagValue(
    args: string[],
    flag: string,
    defaultValue: string
): string {
    const idx = args.indexOf(flag);
    if (
        idx !== -1 &&
        idx + 1 < args.length &&
        !args[idx + 1].startsWith("--")
    ) {
        return args[idx + 1];
    }
    return defaultValue;
}

function parseArgs(argv: string[]) {
    const args = argv.slice(2);
    return {
        dryRun: args.includes("--dry-run"),
        clear: args.includes("--clear"),
        dilemmasPath: getFlagValue(
            args,
            "--dilemmas",
            path.join(BATTERY_ROOT, "dilemmas", "all_dilemmas.json")
        ),
        resultPaths: getResultPaths(args),
    };
}

function getResultPaths(args: string[]): string[] {
    if (args.includes("--results")) {
        const idx = args.indexOf("--results");
        const paths: string[] = [];
        for (
            let i = idx + 1;
            i < args.length && !args[i].startsWith("--");
            i++
        ) {
            paths.push(args[i]);
        }
        return paths;
    }

    // Auto-discover: find latest result file per target model
    const resultsDir = path.join(BATTERY_ROOT, "results");
    if (!fs.existsSync(resultsDir)) {
        console.warn("Results directory not found:", resultsDir);
        return [];
    }

    const allFiles = fs
        .readdirSync(resultsDir)
        .filter((f) => f.startsWith("run_") && f.endsWith(".json") && !f.includes(".scored"))
        .sort()
        .reverse();

    const latest: Map<string, string> = new Map();
    for (const file of allFiles) {
        for (const key of TARGET_MODEL_KEYS) {
            if (!latest.has(key) && file.includes(key)) {
                latest.set(key, path.join(resultsDir, file));
            }
        }
        if (latest.size === TARGET_MODEL_KEYS.length) break;
    }

    const paths = [...latest.values()];
    if (paths.length === 0) {
        console.warn("No result files found for target models");
    }
    return paths;
}

const CREATE_TABLES_SQL = `
    CREATE TABLE IF NOT EXISTS dilemma_sets (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS dilemmas (
        id TEXT PRIMARY KEY,
        set_id TEXT REFERENCES dilemma_sets(id),
        category TEXT NOT NULL,
        title TEXT NOT NULL,
        scenario TEXT NOT NULL,
        question TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS model_responses (
        id TEXT PRIMARY KEY,
        dilemma_id TEXT REFERENCES dilemmas(id) NOT NULL,
        model_key TEXT NOT NULL,
        response_text TEXT NOT NULL,
        source_run_id TEXT,
        UNIQUE(dilemma_id, model_key)
    );
    CREATE TABLE IF NOT EXISTS evaluators (
        id TEXT PRIMARY KEY,
        display_name TEXT NOT NULL DEFAULT 'Anonymous',
        created_at TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE TABLE IF NOT EXISTS evaluations (
        id TEXT PRIMARY KEY,
        evaluator_id TEXT REFERENCES evaluators(id) NOT NULL,
        dilemma_id TEXT REFERENCES dilemmas(id) NOT NULL,
        rank_1_model TEXT NOT NULL,
        rank_2_model TEXT NOT NULL,
        rank_3_model TEXT NOT NULL,
        blind_mapping TEXT NOT NULL,
        presentation_order TEXT NOT NULL,
        rationale TEXT,
        elapsed_seconds REAL,
        created_at TEXT NOT NULL DEFAULT (datetime('now')),
        UNIQUE(evaluator_id, dilemma_id)
    );
    CREATE TABLE IF NOT EXISTS assignments (
        evaluator_id TEXT REFERENCES evaluators(id) NOT NULL,
        dilemma_id TEXT REFERENCES dilemmas(id) NOT NULL,
        assigned_at TEXT NOT NULL DEFAULT (datetime('now')),
        completed INTEGER NOT NULL DEFAULT 0,
        UNIQUE(evaluator_id, dilemma_id)
    );
`;

async function main() {
    const { dryRun, clear, dilemmasPath, resultPaths } = parseArgs(
        process.argv
    );

    if (dryRun) {
        console.log("[DRY RUN] No data will be written.\n");
    }

    // Validate inputs
    if (!fs.existsSync(dilemmasPath)) {
        console.error("Dilemmas file not found:", dilemmasPath);
        process.exit(1);
    }

    for (const rp of resultPaths) {
        if (!fs.existsSync(rp)) {
            console.error("Result file not found:", rp);
            process.exit(1);
        }
    }

    // Read data first (before touching DB)
    const dilemmaData: DilemmaFile = JSON.parse(
        fs.readFileSync(dilemmasPath, "utf-8")
    );
    const resultDataList: Array<{ path: string; data: ResultFile }> = [];
    for (const rp of resultPaths) {
        try {
            const data: ResultFile = JSON.parse(
                fs.readFileSync(rp, "utf-8")
            );
            if (!data.model_key) {
                console.warn(`Skipping ${path.basename(rp)}: no model_key`);
                continue;
            }
            resultDataList.push({ path: rp, data });
        } catch (err) {
            console.error(`Error reading ${rp}:`, err);
            process.exit(1);
        }
    }

    console.log(
        `Dilemmas file: ${path.basename(dilemmasPath)} (${dilemmaData.dilemmas.length} dilemmas)`
    );
    console.log(
        `Result files: ${resultDataList.map((r) => path.basename(r.path)).join(", ") || "(none)"}`
    );
    console.log();

    if (dryRun) {
        const modelKeys = [
            ...new Set(resultDataList.map((r) => r.data.model_key)),
        ];
        const totalResponses = resultDataList.reduce(
            (sum, r) => sum + (r.data.responses?.length ?? 0),
            0
        );
        console.log("=== Dry Run Summary ===");
        console.log(
            `Would import ${dilemmaData.dilemmas.length} dilemmas`
        );
        console.log(
            `Would import ${totalResponses} responses across ${modelKeys.length} models (${modelKeys.join(", ")})`
        );
        return;
    }

    // Open DB
    fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
    const sqlite = new Database(DB_PATH);
    sqlite.pragma("journal_mode = WAL");
    sqlite.pragma("foreign_keys = ON");
    const db = drizzle(sqlite, { schema });

    // Create tables
    sqlite.exec(CREATE_TABLES_SQL);

    // Wrap all writes in a transaction for atomicity.
    // Buffer log messages so they only print after successful commit.
    const modelKeysImported = new Set<string>();
    let dilemmaCount = 0;
    let responseCount = 0;
    const logBuffer: string[] = [];

    const runImport = sqlite.transaction(() => {
        // Clear if requested
        if (clear) {
            sqlite.exec(`
                DELETE FROM assignments;
                DELETE FROM evaluations;
                DELETE FROM model_responses;
                DELETE FROM dilemmas;
                DELETE FROM dilemma_sets;
                DELETE FROM evaluators;
            `);
            logBuffer.push("Cleared existing data.");
        }

        // Import dilemmas
        const setId = uuid();
        const setName = `${dilemmaData.description || "Battery"} v${dilemmaData.version || "1"}`;

        db.insert(schema.dilemmaSets)
            .values({ id: setId, name: setName })
            .onConflictDoNothing()
            .run();

        const dilemmaValues = dilemmaData.dilemmas.map((d) => ({
            id: d.id,
            setId,
            category: d.category,
            title: d.title,
            scenario: d.scenario,
            question: d.question,
        }));
        if (dilemmaValues.length > 0) {
            db.insert(schema.dilemmas)
                .values(dilemmaValues)
                .onConflictDoNothing()
                .run();
        }
        dilemmaCount = dilemmaValues.length;
        logBuffer.push(`Imported ${dilemmaCount} dilemmas`);

        // Import model responses (bulk per file)
        for (const { path: resultPath, data: runData } of resultDataList) {
            const modelKey = runData.model_key;
            const runId = runData.run_id;
            modelKeysImported.add(modelKey);

            const responsesToInsert = (runData.responses ?? [])
                .filter((resp) => resp.dilemma_id && resp.response)
                .map((resp) => ({
                    id: uuid(),
                    dilemmaId: resp.dilemma_id,
                    modelKey,
                    responseText: resp.response,
                    sourceRunId: runId,
                }));

            const fileResponses = responsesToInsert.length;
            if (fileResponses > 0) {
                db.insert(schema.modelResponses)
                    .values(responsesToInsert)
                    .onConflictDoNothing()
                    .run();
                responseCount += fileResponses;
            }
            logBuffer.push(
                `Imported ${fileResponses} responses from ${path.basename(resultPath)} (${modelKey})`
            );
        }
    });

    runImport();

    // Print buffered logs only after successful commit
    for (const msg of logBuffer) {
        console.log(msg);
    }

    console.log(
        `\n=== Summary ===`
    );
    console.log(
        `${dilemmaCount} dilemmas imported, ${responseCount} responses imported across ${modelKeysImported.size} models`
    );
    console.log(`Database: ${DB_PATH}`);
    sqlite.close();
}

main().catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
});
