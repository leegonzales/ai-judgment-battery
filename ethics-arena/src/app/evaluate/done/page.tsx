"use client";

import Link from "next/link";

export default function DonePage() {
    return (
        <div className="flex min-h-screen items-center justify-center bg-gray-950 px-4">
            <div className="text-center">
                <h1 className="mb-4 text-3xl font-bold text-gray-100">
                    All Done!
                </h1>
                <p className="mb-8 text-gray-400">
                    Thank you for completing your evaluations. Your rankings
                    have been recorded.
                </p>
                <Link
                    href="/"
                    className="rounded-lg bg-indigo-600 px-6 py-3 text-sm font-semibold text-white hover:bg-indigo-500"
                >
                    Back to Home
                </Link>
            </div>
        </div>
    );
}
