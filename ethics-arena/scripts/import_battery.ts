/**
 * Import dilemmas and model responses from the AI Judgment Battery data.
 *
 * Usage: npx tsx scripts/import_battery.ts [--dilemmas path] [--results path1 path2 ...]
 *
 * Defaults to ../dilemmas/all_dilemmas.json and all ../results/run_*.json files.
 */

import Database from "better-sqlite3";
import { drizzle } from "drizzle-orm/better-sqlite3";
import * as schema from "../src/db/schema";
import fs from "fs";
import path from "path";
import { v4 as uuid } from "uuid";
import { glob } from "fs/promises";

const DB_PATH = path.join(__dirname, "..", "data", "ethics-arena.db");
const BATTERY_ROOT = path.join(__dirname, "..", "..");

async function main() {
  // Parse args
  const args = process.argv.slice(2);
  const dilemmasPath =
    args.includes("--dilemmas")
      ? args[args.indexOf("--dilemmas") + 1]
      : path.join(BATTERY_ROOT, "dilemmas", "all_dilemmas.json");

  let resultPaths: string[] = [];
  if (args.includes("--results")) {
    const idx = args.indexOf("--results");
    for (let i = idx + 1; i < args.length && !args[i].startsWith("--"); i++) {
      resultPaths.push(args[i]);
    }
  } else {
    // Auto-discover result files
    const resultsDir = path.join(BATTERY_ROOT, "results");
    if (fs.existsSync(resultsDir)) {
      const files = fs.readdirSync(resultsDir).filter((f) => f.startsWith("run_") && f.endsWith(".json"));
      resultPaths = files.map((f) => path.join(resultsDir, f));
    }
  }

  // Open DB
  fs.mkdirSync(path.dirname(DB_PATH), { recursive: true });
  const sqlite = new Database(DB_PATH);
  sqlite.pragma("journal_mode = WAL");
  sqlite.pragma("foreign_keys = ON");
  const db = drizzle(sqlite, { schema });

  // Create tables
  sqlite.exec(`
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
  `);

  // Import dilemmas
  console.log(`Importing dilemmas from ${dilemmasPath}`);
  const dilemmaData = JSON.parse(fs.readFileSync(dilemmasPath, "utf-8"));
  const setId = uuid();
  const setName = `${dilemmaData.description || "Battery"} v${dilemmaData.version || "1"}`;

  db.insert(schema.dilemmaSets)
    .values({ id: setId, name: setName })
    .onConflictDoNothing()
    .run();

  let dilemmaCount = 0;
  for (const d of dilemmaData.dilemmas) {
    db.insert(schema.dilemmas)
      .values({
        id: d.id,
        setId,
        category: d.category,
        title: d.title,
        scenario: d.scenario,
        question: d.question,
      })
      .onConflictDoNothing()
      .run();
    dilemmaCount++;
  }
  console.log(`  Imported ${dilemmaCount} dilemmas`);

  // Import model responses
  // We want exactly 3 models. Let the user pick or auto-select the latest runs.
  // For now, import all discovered result files.
  const modelKeysImported = new Set<string>();
  let responseCount = 0;

  for (const resultPath of resultPaths) {
    console.log(`Importing responses from ${path.basename(resultPath)}`);
    const runData = JSON.parse(fs.readFileSync(resultPath, "utf-8"));
    const modelKey = runData.model_key;
    const runId = runData.run_id;

    if (!modelKey) {
      console.log(`  Skipping (no model_key)`);
      continue;
    }

    modelKeysImported.add(modelKey);

    for (const resp of runData.responses || []) {
      db.insert(schema.modelResponses)
        .values({
          id: uuid(),
          dilemmaId: resp.dilemma_id,
          modelKey,
          responseText: resp.response,
          sourceRunId: runId,
        })
        .onConflictDoNothing()
        .run();
      responseCount++;
    }
  }

  console.log(`  Imported ${responseCount} responses from models: ${[...modelKeysImported].join(", ")}`);
  console.log("\nDone! Database at:", DB_PATH);
  sqlite.close();
}

main().catch(console.error);
