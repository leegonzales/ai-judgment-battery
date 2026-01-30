import { NextRequest, NextResponse } from "next/server";
import { db, schema } from "@/db";
import { eq, and, sql } from "drizzle-orm";

/**
 * GET /api/evaluate/progress?evaluator_id=X
 * Returns evaluation progress for an evaluator.
 */
export async function GET(request: NextRequest) {
    const evaluatorId = request.nextUrl.searchParams.get("evaluator_id");
    if (!evaluatorId) {
        return NextResponse.json(
            { error: "evaluator_id is required" },
            { status: 400 }
        );
    }

    // Count total assignments and completed
    const stats = db
        .select({
            total: sql<number>`count(*)`,
            completed: sql<number>`sum(case when completed = 1 then 1 else 0 end)`,
        })
        .from(schema.assignments)
        .where(eq(schema.assignments.evaluatorId, evaluatorId))
        .get();

    const total = stats?.total ?? 0;
    const completed = stats?.completed ?? 0;

    // Average elapsed time for this evaluator
    const timeStats = db
        .select({
            avgTime: sql<number>`avg(elapsed_seconds)`,
        })
        .from(schema.evaluations)
        .where(eq(schema.evaluations.evaluatorId, evaluatorId))
        .get();

    return NextResponse.json({
        completed,
        total,
        remaining: total - completed,
        stats: {
            avg_elapsed_seconds: timeStats?.avgTime
                ? +timeStats.avgTime.toFixed(1)
                : null,
        },
    });
}
