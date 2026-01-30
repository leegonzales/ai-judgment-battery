import { NextRequest, NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import { db, schema } from "@/db";
import { eq, and } from "drizzle-orm";
import { generateBlinding } from "@/lib/blinding";

interface RankingEntry {
    label: string;
    rank: number;
}

/**
 * POST /api/evaluate/submit
 * Accepts ranking submission and stores evaluation.
 * Body: { evaluator_id, dilemma_id, rankings: [{label, rank}], rationale?, elapsed_seconds? }
 */
export async function POST(request: NextRequest) {
    let body: {
        evaluator_id: string;
        dilemma_id: string;
        rankings: RankingEntry[];
        rationale?: string;
        elapsed_seconds?: number;
    };

    try {
        body = await request.json();
    } catch {
        return NextResponse.json(
            { error: "Invalid JSON body" },
            { status: 400 }
        );
    }

    const { evaluator_id, dilemma_id, rankings, rationale, elapsed_seconds } = body;

    if (!evaluator_id || !dilemma_id || !rankings || !Array.isArray(rankings)) {
        return NextResponse.json(
            { error: "evaluator_id, dilemma_id, and rankings are required" },
            { status: 400 }
        );
    }

    if (rankings.length !== 3) {
        return NextResponse.json(
            { error: "Exactly 3 rankings required" },
            { status: 400 }
        );
    }

    // Validate ranks are 1, 2, 3
    const ranks = rankings.map((r) => r.rank).sort();
    if (ranks[0] !== 1 || ranks[1] !== 2 || ranks[2] !== 3) {
        return NextResponse.json(
            { error: "Rankings must include ranks 1, 2, and 3" },
            { status: 400 }
        );
    }

    // Get model responses to reconstruct blinding
    const modelResponses = db
        .select()
        .from(schema.modelResponses)
        .where(eq(schema.modelResponses.dilemmaId, dilemma_id))
        .all();

    if (modelResponses.length === 0) {
        return NextResponse.json(
            { error: "No model responses found for this dilemma" },
            { status: 404 }
        );
    }

    const modelKeys = modelResponses.map((r) => r.modelKey).sort();
    const blinding = generateBlinding(evaluator_id, dilemma_id, modelKeys);

    // Convert label rankings to model rankings
    const sortedRankings = [...rankings].sort((a, b) => a.rank - b.rank);
    const rank1Model = blinding.labelToModel[sortedRankings[0].label];
    const rank2Model = blinding.labelToModel[sortedRankings[1].label];
    const rank3Model = blinding.labelToModel[sortedRankings[2].label];

    if (!rank1Model || !rank2Model || !rank3Model) {
        return NextResponse.json(
            { error: "Invalid labels in rankings" },
            { status: 400 }
        );
    }

    // Store evaluation
    try {
        db.insert(schema.evaluations)
            .values({
                id: uuidv4(),
                evaluatorId: evaluator_id,
                dilemmaId: dilemma_id,
                rank1Model,
                rank2Model,
                rank3Model,
                blindMapping: JSON.stringify(blinding.labelToModel),
                presentationOrder: JSON.stringify(blinding.presentationOrder),
                rationale: rationale || null,
                elapsedSeconds: elapsed_seconds || null,
            })
            .run();
    } catch (err: unknown) {
        const message = err instanceof Error ? err.message : String(err);
        if (message.includes("UNIQUE constraint")) {
            return NextResponse.json(
                { error: "Already evaluated this dilemma" },
                { status: 409 }
            );
        }
        throw err;
    }

    // Mark assignment completed
    db.update(schema.assignments)
        .set({ completed: 1 })
        .where(
            and(
                eq(schema.assignments.evaluatorId, evaluator_id),
                eq(schema.assignments.dilemmaId, dilemma_id)
            )
        )
        .run();

    return NextResponse.json({ success: true });
}
