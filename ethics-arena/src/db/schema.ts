import { sqliteTable, text, integer, real, uniqueIndex } from "drizzle-orm/sqlite-core";

export const dilemmaSets = sqliteTable("dilemma_sets", {
  id: text("id").primaryKey(),
  name: text("name").notNull(),
  createdAt: text("created_at").notNull().default("(datetime('now'))"),
});

export const dilemmas = sqliteTable("dilemmas", {
  id: text("id").primaryKey(),
  setId: text("set_id").references(() => dilemmaSets.id),
  category: text("category").notNull(),
  title: text("title").notNull(),
  scenario: text("scenario").notNull(),
  question: text("question").notNull(),
});

export const modelResponses = sqliteTable(
  "model_responses",
  {
    id: text("id").primaryKey(),
    dilemmaId: text("dilemma_id")
      .references(() => dilemmas.id)
      .notNull(),
    modelKey: text("model_key").notNull(),
    responseText: text("response_text").notNull(),
    sourceRunId: text("source_run_id"),
  },
  (table) => [
    uniqueIndex("model_responses_dilemma_model").on(
      table.dilemmaId,
      table.modelKey
    ),
  ]
);

export const evaluators = sqliteTable("evaluators", {
  id: text("id").primaryKey(),
  displayName: text("display_name").notNull().default("Anonymous"),
  createdAt: text("created_at").notNull().default("(datetime('now'))"),
});

export const evaluations = sqliteTable(
  "evaluations",
  {
    id: text("id").primaryKey(),
    evaluatorId: text("evaluator_id")
      .references(() => evaluators.id)
      .notNull(),
    dilemmaId: text("dilemma_id")
      .references(() => dilemmas.id)
      .notNull(),
    rank1Model: text("rank_1_model").notNull(),
    rank2Model: text("rank_2_model").notNull(),
    rank3Model: text("rank_3_model").notNull(),
    blindMapping: text("blind_mapping").notNull(), // JSON
    presentationOrder: text("presentation_order").notNull(), // JSON
    rationale: text("rationale"),
    elapsedSeconds: real("elapsed_seconds"),
    createdAt: text("created_at").notNull().default("(datetime('now'))"),
  },
  (table) => [
    uniqueIndex("evaluations_evaluator_dilemma").on(
      table.evaluatorId,
      table.dilemmaId
    ),
  ]
);

export const assignments = sqliteTable(
  "assignments",
  {
    evaluatorId: text("evaluator_id")
      .references(() => evaluators.id)
      .notNull(),
    dilemmaId: text("dilemma_id")
      .references(() => dilemmas.id)
      .notNull(),
    assignedAt: text("assigned_at").notNull().default("(datetime('now'))"),
    completed: integer("completed").notNull().default(0),
  },
  (table) => [
    uniqueIndex("assignments_pk").on(table.evaluatorId, table.dilemmaId),
  ]
);
