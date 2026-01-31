"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

export default function EvaluateSetup() {
    const router = useRouter();
    const [displayName, setDisplayName] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    async function handleBegin() {
        setLoading(true);
        setError("");

        try {
            const res = await fetch("/api/session/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    display_name: displayName.trim() || undefined,
                }),
            });

            if (!res.ok) {
                throw new Error("Failed to start session");
            }

            const data = await res.json();
            localStorage.setItem("evaluator_id", data.evaluator_id);

            // Fetch first dilemma to get its ID for redirect
            const nextRes = await fetch(
                `/api/evaluate/next?evaluator_id=${data.evaluator_id}`
            );
            const nextData = await nextRes.json();

            if (nextData.done) {
                setError("No dilemmas available right now.");
                setLoading(false);
                return;
            }

            router.push(`/evaluate/${nextData.dilemma.id}`);
        } catch {
            setError("Something went wrong. Please try again.");
            setLoading(false);
        }
    }

    return (
        <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 text-white">
            <div className="w-full max-w-md space-y-8">
                <div className="text-center">
                    <h1 className="text-3xl font-bold tracking-tight">
                        Welcome to{" "}
                        <span className="text-indigo-400">
                            EthicsArena
                        </span>
                    </h1>
                    <p className="mt-3 text-gray-400">
                        Set up your evaluator profile
                    </p>
                </div>

                <div className="rounded-xl border border-gray-800 bg-gray-900 p-6 space-y-6">
                    <div>
                        <label
                            htmlFor="displayName"
                            className="block text-sm font-medium text-gray-300"
                        >
                            Display Name
                        </label>
                        <input
                            id="displayName"
                            type="text"
                            placeholder="Anonymous"
                            value={displayName}
                            onChange={(e) =>
                                setDisplayName(e.target.value)
                            }
                            className="mt-2 w-full rounded-lg border border-gray-700 bg-gray-800 px-4 py-2.5 text-white placeholder-gray-500 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                        />
                    </div>

                    <div className="rounded-lg bg-gray-800/50 p-4 text-sm leading-relaxed text-gray-400">
                        You will see ethical dilemmas with 3
                        AI-generated responses. Rank them 1st, 2nd, 3rd.
                        Model names are hidden.
                    </div>

                    {error && (
                        <p className="text-sm text-red-400">{error}</p>
                    )}

                    <button
                        onClick={handleBegin}
                        disabled={loading}
                        className="w-full rounded-lg bg-indigo-600 py-3 font-semibold transition-colors hover:bg-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed"
                    >
                        {loading ? "Starting..." : "Begin"}
                    </button>
                </div>
            </div>
        </div>
    );
}
