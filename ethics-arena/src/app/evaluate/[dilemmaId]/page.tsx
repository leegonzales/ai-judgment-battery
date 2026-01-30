"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useRouter } from "next/navigation";
import DilemmaCard from "@/components/DilemmaCard";
import ResponsePanel from "@/components/ResponsePanel";
import RankingInterface from "@/components/RankingInterface";
import ProgressBar from "@/components/ProgressBar";

interface Dilemma {
    id: string;
    title: string;
    scenario: string;
    question: string;
}

interface ResponseItem {
    label: string;
    text: string;
}

interface NextResponse {
    done?: boolean;
    dilemma: Dilemma;
    responses: ResponseItem[];
    presentation_order: string[];
}

interface ProgressData {
    completed: number;
    total: number;
}

function getEvaluatorId(): string | null {
    if (typeof localStorage === "undefined") return null;
    try {
        let id = localStorage.getItem("evaluator_id");
        if (!id) {
            id = crypto.randomUUID();
            localStorage.setItem("evaluator_id", id);
        }
        return id;
    } catch (err) {
        console.error("Error accessing localStorage:", err);
        return null;
    }
}

export default function EvaluatePage() {
    const router = useRouter();
    const [dilemma, setDilemma] = useState<Dilemma | null>(null);
    const [responses, setResponses] = useState<ResponseItem[]>([]);
    const [labels, setLabels] = useState<string[]>([]);
    const [rankings, setRankings] = useState<Record<string, number | null>>(
        {}
    );
    const [progress, setProgress] = useState<ProgressData>({
        completed: 0,
        total: 0,
    });
    const [submitting, setSubmitting] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const startTime = useRef(Date.now());

    const fetchNext = useCallback(async () => {
        setLoading(true);
        setError(null);
        const evaluatorId = getEvaluatorId();
        if (!evaluatorId) {
            setError("Unable to access local storage. Please enable it and reload.");
            setLoading(false);
            return;
        }

        try {
            const [nextRes, progRes] = await Promise.all([
                fetch(
                    `/api/evaluate/next?evaluator_id=${encodeURIComponent(evaluatorId)}`
                ),
                fetch(
                    `/api/evaluate/progress?evaluator_id=${encodeURIComponent(evaluatorId)}`
                ),
            ]);

            const nextData: NextResponse = await nextRes.json();
            const progData: ProgressData = await progRes.json();

            setProgress(progData);

            if (nextData.done) {
                router.push("/evaluate/done");
                return;
            }

            setDilemma(nextData.dilemma);
            setResponses(nextData.responses);
            setLabels(nextData.presentation_order);
            setRankings(
                Object.fromEntries(
                    nextData.presentation_order.map((l) => [l, null])
                )
            );
            startTime.current = Date.now();

            // Update URL without navigation
            window.history.replaceState(
                null,
                "",
                `/evaluate/${nextData.dilemma.id}`
            );
        } catch (err) {
            console.error("Failed to load dilemma:", err);
            setError("Failed to load dilemma. Please try again.");
        } finally {
            setLoading(false);
        }
    }, [router]);

    useEffect(() => {
        fetchNext();
    }, [fetchNext]);

    function handleSetRank(label: string, rank: number) {
        setRankings((prev) => {
            const next = { ...prev };
            if (prev[label] === rank) {
                next[label] = null;
            } else {
                for (const k of Object.keys(next)) {
                    if (next[k] === rank) next[k] = null;
                }
                next[label] = rank;
            }
            return next;
        });
    }

    function handleClear() {
        setRankings(Object.fromEntries(labels.map((l) => [l, null])));
    }

    // Click a response panel to toggle or assign next available rank
    function handleResponseClick(label: string) {
        setRankings((prev) => {
            if (prev[label] !== null) {
                return { ...prev, [label]: null };
            }
            const assigned = new Set(
                Object.values(prev).filter((v): v is number => v !== null)
            );
            for (const rank of [1, 2, 3]) {
                if (!assigned.has(rank)) {
                    return { ...prev, [label]: rank };
                }
            }
            return prev;
        });
    }

    async function handleSubmit(rationale: string) {
        if (!dilemma) return;
        setSubmitting(true);
        const evaluatorId = getEvaluatorId();
        if (!evaluatorId) {
            setError("Unable to access local storage.");
            setSubmitting(false);
            return;
        }
        const elapsed = Math.round((Date.now() - startTime.current) / 1000);

        try {
            const res = await fetch("/api/evaluate/submit", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    evaluator_id: evaluatorId,
                    dilemma_id: dilemma.id,
                    rankings: labels.map((label) => ({
                        label,
                        rank: rankings[label],
                    })),
                    rationale: rationale || undefined,
                    elapsed_seconds: elapsed,
                }),
            });

            if (!res.ok) {
                try {
                    const data = await res.json();
                    setError(data.error || "Submission failed");
                } catch {
                    setError(
                        `Submission failed with status: ${res.status}`
                    );
                }
                return;
            }

            await fetchNext();
        } catch (err) {
            console.error("Submission failed:", err);
            setError("Submission failed. Please try again.");
        } finally {
            setSubmitting(false);
        }
    }

    function handleSkip() {
        fetchNext();
    }

    if (loading) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-gray-950">
                <p className="text-gray-400">Loading...</p>
            </div>
        );
    }

    if (error && !dilemma) {
        return (
            <div className="flex min-h-screen items-center justify-center bg-gray-950">
                <div className="text-center">
                    <p className="mb-4 text-red-400">{error}</p>
                    <button
                        type="button"
                        onClick={fetchNext}
                        className="rounded-lg bg-indigo-600 px-4 py-2 text-sm text-white hover:bg-indigo-500"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    if (!dilemma) return null;

    return (
        <div className="min-h-screen bg-gray-950 px-4 py-8">
            <div className="mx-auto max-w-3xl space-y-6">
                <ProgressBar
                    completed={progress.completed}
                    total={progress.total}
                />

                {error && (
                    <p className="text-center text-sm text-red-400">{error}</p>
                )}

                <DilemmaCard
                    title={dilemma.title}
                    scenario={dilemma.scenario}
                    question={dilemma.question}
                />

                <div className="space-y-4">
                    {responses.map((r) => (
                        <ResponsePanel
                            key={r.label}
                            label={r.label}
                            text={r.text}
                            rank={rankings[r.label]}
                            onClick={() => handleResponseClick(r.label)}
                        />
                    ))}
                </div>

                <RankingInterface
                    labels={labels}
                    rankings={rankings}
                    onSetRank={handleSetRank}
                    onClear={handleClear}
                    onSubmit={handleSubmit}
                    submitting={submitting}
                    onSkip={handleSkip}
                />
            </div>
        </div>
    );
}
