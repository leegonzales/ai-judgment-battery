import { NextRequest, NextResponse } from "next/server";
import { requireAdmin } from "@/lib/auth";
import { exportResults } from "@/lib/export";

/**
 * GET /api/admin/export
 * Returns pipeline-compatible JSON export. Requires admin auth.
 */
export async function GET(request: NextRequest) {
    const authError = requireAdmin(request);
    if (authError) return authError;

    const data = exportResults();
    return NextResponse.json(data);
}
