"use client";

import { useState } from "react";

const RANK_LABELS: Record<number, string> = { 1: "1st", 2: "2nd", 3: "3rd" };

function rankButtonClass(isActive: boolean, isTaken: boolean): string {
    const base = "rounded-lg px-3 py-1 text-sm font-medium transition-colors";
    if (isActive) return `${base} bg-indigo-600 text-white`;
    if (isTaken) return `${base} cursor-not-allowed bg-gray-800 text-gray-600`;
    return `${base} bg-gray-800 text-gray-300 hover:bg-gray-700`;
}

interface RankingInterfaceProps {
    labels: string[];
    rankings: Record<string, number | null>;
    onSetRank: (label: string, rank: number) => void;
    onClear: () => void;
    onSubmit: (rationale: string) => void;
    submitting: boolean;
    onSkip: () => void;
}

export default function RankingInterface({
    labels,
    rankings,
    onSetRank,
    onClear,
    onSubmit,
    submitting,
    onSkip,
}: RankingInterfaceProps) {
    const [rationale, setRationale] = useState("");

    const assignedRanks = new Set(
        Object.values(rankings).filter((v): v is number => v !== null)
    );
    const allRanked = labels.every((l) => rankings[l] !== null);
    const canSubmit = allRanked && !submitting;

    return (
        <div className="space-y-4">
            <div className="rounded-xl border border-gray-700 bg-gray-900 p-5">
                <div className="mb-3 flex items-center justify-between">
                    <h3 className="text-sm font-semibold text-gray-400">
                        Assign Rankings
                    </h3>
                    <button
                        type="button"
                        onClick={onClear}
                        className="text-xs text-gray-500 hover:text-gray-300"
                    >
                        Clear
                    </button>
                </div>
                <div className="space-y-3">
                    {labels.map((label) => (
                        <div key={label} className="flex items-center gap-3">
                            <span className="w-28 text-sm text-gray-300">
                                Response {label}
                            </span>
                            {[1, 2, 3].map((rank) => {
                                const isActive = rankings[label] === rank;
                                const isTaken =
                                    !isActive && assignedRanks.has(rank);
                                return (
                                    <button
                                        key={rank}
                                        type="button"
                                        disabled={isTaken}
                                        onClick={() => onSetRank(label, rank)}
                                        className={rankButtonClass(isActive, isTaken)}
                                    >
                                        {RANK_LABELS[rank]}
                                    </button>
                                );
                            })}
                        </div>
                    ))}
                </div>
            </div>

            <textarea
                value={rationale}
                onChange={(e) => setRationale(e.target.value)}
                placeholder="Optional: explain your reasoning..."
                className="w-full rounded-xl border border-gray-700 bg-gray-900 p-4 text-sm text-gray-300 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
                rows={3}
            />

            <div className="flex items-center justify-between">
                <button
                    type="button"
                    onClick={onSkip}
                    className="text-sm text-gray-500 hover:text-gray-300"
                >
                    Skip
                </button>
                <button
                    type="button"
                    disabled={!canSubmit}
                    onClick={() => onSubmit(rationale)}
                    className={`rounded-lg px-6 py-2.5 text-sm font-semibold transition-colors ${
                        canSubmit
                            ? "bg-indigo-600 text-white hover:bg-indigo-500"
                            : "cursor-not-allowed bg-gray-800 text-gray-600"
                    }`}
                >
                    {submitting ? "Submitting..." : "Submit Rankings"}
                </button>
            </div>
        </div>
    );
}
