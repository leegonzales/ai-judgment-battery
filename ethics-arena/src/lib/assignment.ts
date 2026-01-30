/**
 * Dilemma assignment algorithm.
 * Prioritizes under-evaluated dilemmas with category stratification.
 */

import { db, schema } from "@/db";
import { eq, sql, and, notInArray } from "drizzle-orm";

export interface AssignmentConfig {
  sessionSize: number; // default 10
  minCoverage: number; // minimum evaluations per dilemma, default 3
}

const DEFAULT_CONFIG: AssignmentConfig = {
  sessionSize: 10,
  minCoverage: 3,
};

/**
 * Get the next batch of dilemmas for an evaluator.
 * Algorithm:
 * 1. Find dilemmas with fewest human evaluations
 * 2. Exclude already-completed ones
 * 3. Stratify by category for diversity
 * 4. Return up to sessionSize
 */
export async function assignDilemmas(
  evaluatorId: string,
  config: Partial<AssignmentConfig> = {}
): Promise<string[]> {
  const { sessionSize } = { ...DEFAULT_CONFIG, ...config };

  // Get already completed dilemma IDs for this evaluator
  const completed = db
    .select({ dilemmaId: schema.evaluations.dilemmaId })
    .from(schema.evaluations)
    .where(eq(schema.evaluations.evaluatorId, evaluatorId))
    .all();
  const completedIds = new Set(completed.map((r) => r.dilemmaId));

  // Get evaluation counts per dilemma
  const evalCounts = db
    .select({
      dilemmaId: schema.dilemmas.id,
      category: schema.dilemmas.category,
      evalCount: sql<number>`coalesce((
        select count(*) from evaluations
        where evaluations.dilemma_id = dilemmas.id
      ), 0)`.as("eval_count"),
    })
    .from(schema.dilemmas)
    .all();

  // Filter out completed, sort by eval count (ascending)
  const available = evalCounts
    .filter((d) => !completedIds.has(d.dilemmaId))
    .sort((a, b) => a.evalCount - b.evalCount);

  if (available.length === 0) return [];

  // Stratified sampling: round-robin across categories
  const byCategory = new Map<string, typeof available>();
  for (const d of available) {
    if (!byCategory.has(d.category)) byCategory.set(d.category, []);
    byCategory.get(d.category)!.push(d);
  }

  const result: string[] = [];
  const categories = [...byCategory.keys()].sort();
  let catIdx = 0;

  while (result.length < sessionSize && result.length < available.length) {
    const cat = categories[catIdx % categories.length];
    const items = byCategory.get(cat)!;
    if (items.length > 0) {
      result.push(items.shift()!.dilemmaId);
    }
    catIdx++;
    // Remove empty categories
    if (items.length === 0) {
      byCategory.delete(cat);
      const idx = categories.indexOf(cat);
      if (idx >= 0) categories.splice(idx, 1);
      if (categories.length === 0) break;
    }
  }

  // Create assignments
  for (const dilemmaId of result) {
    db.insert(schema.assignments)
      .values({ evaluatorId, dilemmaId, completed: 0 })
      .onConflictDoNothing()
      .run();
  }

  return result;
}

/** Get existing assigned (uncompleted) dilemmas for an evaluator */
export function getAssignedDilemmas(evaluatorId: string): string[] {
  const rows = db
    .select({ dilemmaId: schema.assignments.dilemmaId })
    .from(schema.assignments)
    .where(
      and(
        eq(schema.assignments.evaluatorId, evaluatorId),
        eq(schema.assignments.completed, 0)
      )
    )
    .all();
  return rows.map((r) => r.dilemmaId);
}
