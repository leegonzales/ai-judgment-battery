import { NextRequest, NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import { db, schema } from "@/db";
import { assignDilemmas } from "@/lib/assignment";

/**
 * GET /api/session/start
 * Creates an anonymous evaluator + session, returns IDs.
 */
export async function GET() {
    const evaluatorId = uuidv4();

    db.insert(schema.evaluators)
        .values({ id: evaluatorId, displayName: "Anonymous" })
        .run();

    const assigned = await assignDilemmas(evaluatorId);

    return NextResponse.json({
        evaluator_id: evaluatorId,
        session_id: uuidv4(),
        assigned_count: assigned.length,
    });
}

/**
 * POST /api/session/start
 * Creates a named evaluator, assigns dilemmas.
 * Body: { display_name?: string }
 */
export async function POST(request: NextRequest) {
    let displayName = "Anonymous";

    try {
        const body = await request.json();
        if (body.display_name && typeof body.display_name === "string") {
            displayName = body.display_name.trim() || "Anonymous";
        }
    } catch {
        // Empty body is fine, use default
    }

    const evaluatorId = uuidv4();

    db.insert(schema.evaluators)
        .values({ id: evaluatorId, displayName })
        .run();

    const assigned = await assignDilemmas(evaluatorId);

    return NextResponse.json({
        evaluator_id: evaluatorId,
        assigned_count: assigned.length,
    });
}
