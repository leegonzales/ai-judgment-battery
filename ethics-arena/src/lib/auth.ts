/**
 * Admin authentication helper.
 * Checks Authorization header against ADMIN_PASSWORD env var.
 */

import { NextRequest, NextResponse } from "next/server";

export function requireAdmin(request: NextRequest): NextResponse | null {
    const password = process.env.ADMIN_PASSWORD;
    if (!password) {
        return NextResponse.json(
            { error: "ADMIN_PASSWORD not configured" },
            { status: 500 }
        );
    }

    const auth = request.headers.get("authorization");
    if (!auth || auth !== `Bearer ${password}`) {
        return NextResponse.json(
            { error: "Unauthorized" },
            { status: 401 }
        );
    }

    return null; // authorized
}
