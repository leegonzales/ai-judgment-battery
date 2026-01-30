import { NextRequest, NextResponse } from "next/server";
import { db, schema } from "@/db";
import { requireAdmin } from "@/lib/auth";

/**
 * GET /api/admin/results
 * Returns all evaluations with details. Requires admin auth.
 */
export async function GET(request: NextRequest) {
    const authError = requireAdmin(request);
    if (authError) return authError;

    const allEvals = db.select().from(schema.evaluations).all();

    const results = allEvals.map((e) => ({
        id: e.id,
        evaluator_id: e.evaluatorId,
        dilemma_id: e.dilemmaId,
        rankings: [
            { rank: 1, model: e.rank1Model },
            { rank: 2, model: e.rank2Model },
            { rank: 3, model: e.rank3Model },
        ],
        blind_mapping: JSON.parse(e.blindMapping),
        presentation_order: JSON.parse(e.presentationOrder),
        rationale: e.rationale,
        elapsed_seconds: e.elapsedSeconds,
        created_at: e.createdAt,
    }));

    return NextResponse.json({
        total: results.length,
        evaluations: results,
    });
}
