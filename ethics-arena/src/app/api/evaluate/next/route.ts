import { NextRequest, NextResponse } from "next/server";
import { db, schema } from "@/db";
import { eq, and } from "drizzle-orm";
import { generateBlinding } from "@/lib/blinding";
import { getAssignedDilemmas } from "@/lib/assignment";

/**
 * GET /api/evaluate/next?evaluator_id=X
 * Returns the next unfinished dilemma with blinded responses.
 */
export async function GET(request: NextRequest) {
    const evaluatorId = request.nextUrl.searchParams.get("evaluator_id");
    if (!evaluatorId) {
        return NextResponse.json(
            { error: "evaluator_id is required" },
            { status: 400 }
        );
    }

    // Get uncompleted assigned dilemmas
    const pendingIds = getAssignedDilemmas(evaluatorId);
    if (pendingIds.length === 0) {
        return NextResponse.json({ done: true, message: "No more dilemmas to evaluate" });
    }

    const dilemmaId = pendingIds[0];

    // Fetch dilemma details
    const dilemma = db
        .select()
        .from(schema.dilemmas)
        .where(eq(schema.dilemmas.id, dilemmaId))
        .get();

    if (!dilemma) {
        return NextResponse.json(
            { error: "Dilemma not found" },
            { status: 404 }
        );
    }

    // Fetch model responses for this dilemma
    const modelResponses = db
        .select()
        .from(schema.modelResponses)
        .where(eq(schema.modelResponses.dilemmaId, dilemmaId))
        .all();

    if (modelResponses.length === 0) {
        return NextResponse.json(
            { error: "No model responses found for this dilemma" },
            { status: 404 }
        );
    }

    const modelKeys = modelResponses.map((r) => r.modelKey).sort();

    // Generate blinding
    const blinding = generateBlinding(evaluatorId, dilemmaId, modelKeys);

    // Build response map: label -> response text
    const responseByModel: Record<string, string> = {};
    for (const r of modelResponses) {
        responseByModel[r.modelKey] = r.responseText;
    }

    // Return responses in presentation order
    const responses = blinding.presentationOrder.map((label) => ({
        label,
        text: responseByModel[blinding.labelToModel[label]],
    }));

    return NextResponse.json({
        dilemma: {
            id: dilemma.id,
            title: dilemma.title,
            scenario: dilemma.scenario,
            question: dilemma.question,
        },
        responses,
        presentation_order: blinding.presentationOrder,
        blind_mapping_id: `${evaluatorId}:${dilemmaId}`,
    });
}
