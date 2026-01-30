/**
 * Blinding & randomization for fair evaluation.
 * Each evaluator+dilemma combo gets a deterministic but unique mapping.
 */

const LABELS = ["A", "B", "C"] as const;

/** Seeded pseudo-random using simple hash */
function seededRandom(seed: string): () => number {
  let h = 0;
  for (let i = 0; i < seed.length; i++) {
    h = ((h << 5) - h + seed.charCodeAt(i)) | 0;
  }
  return () => {
    h = (h * 1664525 + 1013904223) | 0;
    return ((h >>> 0) / 4294967296);
  };
}

function shuffle<T>(arr: T[], rng: () => number): T[] {
  const result = [...arr];
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

export interface BlindMapping {
  /** Label → model_key, e.g. { A: "claude-opus", B: "gpt-5.1", C: "gemini-3-pro" } */
  labelToModel: Record<string, string>;
  /** Presentation order of labels, e.g. ["B", "A", "C"] */
  presentationOrder: string[];
}

/**
 * Generate a deterministic blinding for a given evaluator + dilemma.
 * Same inputs always produce the same mapping (for revisit consistency).
 * Different evaluators get different mappings (for position bias detection).
 */
export function generateBlinding(
  evaluatorId: string,
  dilemmaId: string,
  modelKeys: string[]
): BlindMapping {
  if (modelKeys.length !== 3) {
    throw new Error(`Expected exactly 3 model keys, got ${modelKeys.length}`);
  }

  // Use evaluator+dilemma as seed for label assignment
  const labelRng = seededRandom(`label:${evaluatorId}:${dilemmaId}`);
  const shuffledModels = shuffle(modelKeys, labelRng);

  const labelToModel: Record<string, string> = {};
  LABELS.forEach((label, i) => {
    labelToModel[label] = shuffledModels[i];
  });

  // Use different seed for presentation order
  const orderRng = seededRandom(`order:${evaluatorId}:${dilemmaId}`);
  const presentationOrder = shuffle([...LABELS], orderRng);

  return { labelToModel, presentationOrder };
}

/** Invert mapping: model_key → label */
export function modelToLabel(
  labelToModel: Record<string, string>
): Record<string, string> {
  const result: Record<string, string> = {};
  for (const [label, model] of Object.entries(labelToModel)) {
    result[model] = label;
  }
  return result;
}
