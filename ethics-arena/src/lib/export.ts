/**
 * Export human evaluations in pipeline-compatible format.
 */

import { db, schema } from "@/db";
import { sql } from "drizzle-orm";
import { computeElo, countWins } from "./elo";

export function exportResults() {
  const allEvals = db
    .select()
    .from(schema.evaluations)
    .all();

  const models = new Set<string>();
  const rankings: string[][] = [];
  const comparisons = allEvals.map((e) => {
    const ranking = [e.rank1Model, e.rank2Model, e.rank3Model];
    ranking.forEach((m) => models.add(m));
    rankings.push(ranking);

    return {
      dilemma_id: e.dilemmaId,
      evaluator_id: e.evaluatorId,
      rankings: [
        { rank: 1, model: e.rank1Model },
        { rank: 2, model: e.rank2Model },
        { rank: 3, model: e.rank3Model },
      ],
      rationale: e.rationale || null,
      elapsed_seconds: e.elapsedSeconds || null,
      presentation_order: JSON.parse(e.presentationOrder),
      timestamp: e.createdAt,
    };
  });

  const evaluatorCount = new Set(allEvals.map((e) => e.evaluatorId)).size;
  const dilemmasCovered = new Set(allEvals.map((e) => e.dilemmaId)).size;

  const eloRatings = computeElo(rankings);
  const wins = countWins(rankings);

  // Position bias: how often does label "A" (first in alphabet) win?
  let aWins = 0;
  let total = 0;
  for (const e of allEvals) {
    const mapping = JSON.parse(e.blindMapping);
    const order: string[] = JSON.parse(e.presentationOrder);
    const firstLabel = order[0];
    const firstModel = mapping[firstLabel];
    if (firstModel === e.rank1Model) aWins++;
    total++;
  }

  return {
    type: "human_evaluation",
    source: "ethics-arena",
    timestamp: new Date().toISOString(),
    total_evaluations: allEvals.length,
    unique_evaluators: evaluatorCount,
    dilemmas_covered: dilemmasCovered,
    models_compared: [...models],
    comparisons,
    aggregated: {
      wins,
      elo_ratings: eloRatings,
      position_bias: {
        first_shown_win_rate: total > 0 ? +(aWins / total).toFixed(3) : 0,
      },
    },
  };
}
