import { NextRequest, NextResponse } from "next/server";
import { db, schema } from "@/db";
import { sql } from "drizzle-orm";
import { requireAdmin } from "@/lib/auth";

/**
 * GET /api/admin/dashboard
 * Returns aggregate stats. Requires admin auth.
 */
export async function GET(request: NextRequest) {
    const authError = requireAdmin(request);
    if (authError) return authError;

    const evalStats = db
        .select({
            totalEvaluations: sql<number>`count(*)`,
            uniqueEvaluators: sql<number>`count(distinct evaluator_id)`,
            dilemmasCovered: sql<number>`count(distinct dilemma_id)`,
            avgTime: sql<number>`avg(elapsed_seconds)`,
        })
        .from(schema.evaluations)
        .get();

    const totalDilemmas = db
        .select({ count: sql<number>`count(*)` })
        .from(schema.dilemmas)
        .get();

    return NextResponse.json({
        total_evaluations: evalStats?.totalEvaluations ?? 0,
        unique_evaluators: evalStats?.uniqueEvaluators ?? 0,
        dilemma_coverage: {
            covered: evalStats?.dilemmasCovered ?? 0,
            total: totalDilemmas?.count ?? 0,
        },
        avg_time: evalStats?.avgTime
            ? +evalStats.avgTime.toFixed(1)
            : null,
    });
}
