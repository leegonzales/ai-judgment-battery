import { NextRequest, NextResponse } from "next/server";
import { v4 as uuidv4 } from "uuid";
import { db, schema } from "@/db";
import { requireAdmin } from "@/lib/auth";

interface DilemmaInput {
    id?: string;
    category: string;
    title: string;
    scenario: string;
    question: string;
}

interface ModelResponseInput {
    dilemma_id: string;
    model_key: string;
    response_text: string;
    source_run_id?: string;
}

interface UploadPayload {
    set_name: string;
    dilemmas: DilemmaInput[];
    model_responses: ModelResponseInput[];
}

/**
 * POST /api/admin/upload
 * Accepts dilemma set + model responses JSON upload.
 * Requires admin auth.
 */
export async function POST(request: NextRequest) {
    const authError = requireAdmin(request);
    if (authError) return authError;

    let body: UploadPayload;
    try {
        body = await request.json();
    } catch {
        return NextResponse.json(
            { error: "Invalid JSON body" },
            { status: 400 }
        );
    }

    if (!body.set_name || !body.dilemmas || !body.model_responses) {
        return NextResponse.json(
            { error: "set_name, dilemmas, and model_responses are required" },
            { status: 400 }
        );
    }

    // Create dilemma set
    const setId = uuidv4();
    db.insert(schema.dilemmaSets)
        .values({ id: setId, name: body.set_name })
        .run();

    // Map of provided dilemma ID -> generated ID
    const dilemmaIdMap = new Map<string, string>();

    // Insert dilemmas
    let dilemmaCount = 0;
    for (const d of body.dilemmas) {
        const dilemmaId = d.id || uuidv4();
        dilemmaIdMap.set(d.id || dilemmaId, dilemmaId);

        db.insert(schema.dilemmas)
            .values({
                id: dilemmaId,
                setId,
                category: d.category,
                title: d.title,
                scenario: d.scenario,
                question: d.question,
            })
            .run();
        dilemmaCount++;
    }

    // Insert model responses
    let responseCount = 0;
    for (const r of body.model_responses) {
        const dilemmaId = dilemmaIdMap.get(r.dilemma_id) || r.dilemma_id;

        db.insert(schema.modelResponses)
            .values({
                id: uuidv4(),
                dilemmaId,
                modelKey: r.model_key,
                responseText: r.response_text,
                sourceRunId: r.source_run_id || null,
            })
            .onConflictDoNothing()
            .run();
        responseCount++;
    }

    return NextResponse.json({
        success: true,
        set_id: setId,
        dilemmas_created: dilemmaCount,
        responses_created: responseCount,
    });
}
