/**
 * Elo rating calculations from pairwise comparisons derived from rankings.
 */

const DEFAULT_K = 32;
const DEFAULT_INITIAL = 1500;

export interface EloRatings {
  [modelKey: string]: number;
}

/** Convert a ranking [1st, 2nd, 3rd] into pairwise wins */
function rankingToPairs(
  ranking: string[]
): Array<{ winner: string; loser: string }> {
  const pairs: Array<{ winner: string; loser: string }> = [];
  for (let i = 0; i < ranking.length; i++) {
    for (let j = i + 1; j < ranking.length; j++) {
      pairs.push({ winner: ranking[i], loser: ranking[j] });
    }
  }
  return pairs;
}

/** Expected score for player A vs B */
function expectedScore(ratingA: number, ratingB: number): number {
  return 1 / (1 + Math.pow(10, (ratingB - ratingA) / 400));
}

/** Compute Elo ratings from a list of rankings */
export function computeElo(
  rankings: string[][], // each entry is [1st, 2nd, 3rd] by model_key
  k: number = DEFAULT_K
): EloRatings {
  const ratings: EloRatings = {};

  // Initialize all models
  for (const ranking of rankings) {
    for (const model of ranking) {
      if (!(model in ratings)) ratings[model] = DEFAULT_INITIAL;
    }
  }

  // Process each ranking
  for (const ranking of rankings) {
    const pairs = rankingToPairs(ranking);
    for (const { winner, loser } of pairs) {
      const expectedW = expectedScore(ratings[winner], ratings[loser]);
      const expectedL = expectedScore(ratings[loser], ratings[winner]);
      ratings[winner] += k * (1 - expectedW);
      ratings[loser] += k * (0 - expectedL);
    }
  }

  // Round
  for (const key of Object.keys(ratings)) {
    ratings[key] = Math.round(ratings[key]);
  }

  return ratings;
}

/** Count wins per model from rankings */
export function countWins(
  rankings: string[][]
): Record<string, number> {
  const wins: Record<string, number> = {};
  for (const ranking of rankings) {
    const winner = ranking[0];
    wins[winner] = (wins[winner] || 0) + 1;
  }
  return wins;
}
