"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

interface ProgressData {
  completed: number;
  total: number;
  remaining: number;
  stats: { avg_elapsed_seconds: number | null };
}

export default function DonePage() {
  const router = useRouter();
  const [progress, setProgress] = useState<ProgressData | null>(null);
  const [copied, setCopied] = useState(false);
  const [checking, setChecking] = useState(false);
  const [message, setMessage] = useState("");

  useEffect(() => {
    const evaluatorId = localStorage.getItem("evaluator_id");
    if (!evaluatorId) return;

    fetch(`/api/evaluate/progress?evaluator_id=${evaluatorId}`)
      .then((r) => {
        if (!r.ok) throw new Error("Failed to fetch progress");
        return r.json();
      })
      .then(setProgress)
      .catch((err) => {
        console.error(err);
        setMessage("Could not load session stats.");
      });
  }, []);

  async function handleContinue() {
    const evaluatorId = localStorage.getItem("evaluator_id");
    if (!evaluatorId) return;
    setChecking(true);
    setMessage("");
    try {
      const res = await fetch(
        `/api/evaluate/next?evaluator_id=${evaluatorId}`
      );
      if (!res.ok) {
        throw new Error(`Server error: ${res.statusText}`);
      }
      const data = await res.json();
      if (data.dilemma?.id) {
        router.push(`/evaluate/${data.dilemma.id}`);
        return;
      }
      setMessage("No more dilemmas available right now.");
    } catch (err) {
      console.error(err);
      setMessage("Failed to check for more dilemmas.");
    } finally {
      setChecking(false);
    }
  }

  function handleNewSession() {
    localStorage.removeItem("evaluator_id");
    router.push("/evaluate");
  }

  function handleCopyLink() {
    navigator.clipboard.writeText(window.location.origin).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }).catch((err) => {
      console.error("Could not copy text:", err);
    });
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-950 px-4">
      <div className="w-full max-w-md text-center">
        <h1 className="mb-2 text-3xl font-bold text-gray-100">
          Thank you for evaluating!
        </h1>
        <p className="mb-8 text-gray-400">
          Your rankings have been recorded and will help calibrate AI judgment.
        </p>

        {progress && (
          <div className="mb-8 rounded-lg border border-gray-800 bg-gray-900 p-6">
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-500">
              Your Session
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <div className="text-2xl font-bold text-indigo-400">
                  {progress.completed}
                </div>
                <div className="text-xs text-gray-500">Completed</div>
              </div>
              <div>
                <div className="text-2xl font-bold text-gray-300">
                  {progress.total}
                </div>
                <div className="text-xs text-gray-500">Assigned</div>
              </div>
            </div>
            {progress.stats.avg_elapsed_seconds && (
              <div className="mt-4 text-sm text-gray-500">
                Avg time per evaluation:{" "}
                <span className="text-gray-300">
                  {Math.round(progress.stats.avg_elapsed_seconds)}s
                </span>
              </div>
            )}
          </div>
        )}

        {message && (
          <p className="mb-4 text-sm text-yellow-400">{message}</p>
        )}

        <div className="space-y-3">
          {progress && progress.remaining > 0 && (
            <button
              onClick={handleContinue}
              disabled={checking}
              className="w-full rounded-lg bg-indigo-600 px-6 py-3 text-sm font-semibold text-white hover:bg-indigo-500 disabled:opacity-50"
            >
              {checking
                ? "Checking..."
                : `Continue Evaluating (${progress.remaining} remaining)`}
            </button>
          )}

          <button
            onClick={handleNewSession}
            className="w-full rounded-lg border border-gray-700 px-6 py-3 text-sm font-semibold text-gray-300 hover:border-gray-600 hover:text-white"
          >
            Start New Session
          </button>

          <Link
            href="/"
            className="block w-full rounded-lg px-6 py-3 text-sm text-gray-500 hover:text-gray-400"
          >
            Back to Home
          </Link>
        </div>

        <div className="mt-8 rounded-lg border border-gray-800 bg-gray-900/50 p-4">
          <p className="mb-2 text-xs text-gray-500">
            Invite others to evaluate
          </p>
          <button
            onClick={handleCopyLink}
            className="text-sm text-indigo-400 hover:text-indigo-300"
          >
            {copied ? "Copied!" : "Copy invite link"}
          </button>
        </div>
      </div>
    </div>
  );
}
