"use client";

import { useEffect, useState, useCallback } from "react";
import { WinBars, EloTable } from "@/components/AdminCharts";

interface DashboardData {
  total_evaluations: number;
  unique_evaluators: number;
  dilemma_coverage: { covered: number; total: number };
  avg_time: number | null;
}

interface EvalResult {
  rankings: Array<{ rank: number; model: string }>;
}

interface ResultsData {
  total: number;
  evaluations: EvalResult[];
}

interface AggregatedData {
  wins: Record<string, number>;
  elo_ratings: Record<string, number>;
}

function getAdminHeaders(): HeadersInit {
  const pw = sessionStorage.getItem("admin_password") || "";
  return { Authorization: `Bearer ${pw}` };
}

/** Client-side aggregation: compute wins and Elo from raw evaluations */
function aggregateResults(evaluations: EvalResult[]): AggregatedData {
  const wins: Record<string, number> = {};
  const ratings: Record<string, number> = {};

  for (const ev of evaluations) {
    for (const r of ev.rankings) {
      if (!(r.model in ratings)) ratings[r.model] = 1500;
      if (!(r.model in wins)) wins[r.model] = 0;
    }
    if (ev.rankings.length > 0) {
      const winner = ev.rankings.find((r) => r.rank === 1);
      if (winner) wins[winner.model] = (wins[winner.model] || 0) + 1;
    }

    // Pairwise Elo updates
    const K = 32;
    for (let i = 0; i < ev.rankings.length; i++) {
      for (let j = i + 1; j < ev.rankings.length; j++) {
        const a = ev.rankings[i];
        const b = ev.rankings[j];
        const winner = a.rank < b.rank ? a.model : b.model;
        const loser = a.rank < b.rank ? b.model : a.model;
        const eW = 1 / (1 + Math.pow(10, (ratings[loser] - ratings[winner]) / 400));
        const eL = 1 / (1 + Math.pow(10, (ratings[winner] - ratings[loser]) / 400));
        ratings[winner] += K * (1 - eW);
        ratings[loser] += K * (0 - eL);
      }
    }
  }

  for (const key of Object.keys(ratings)) {
    ratings[key] = Math.round(ratings[key]);
  }

  return { wins, elo_ratings: ratings };
}

export default function AdminDashboard() {
  const [dashboard, setDashboard] = useState<DashboardData | null>(null);
  const [aggregated, setAggregated] = useState<AggregatedData | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadMsg, setUploadMsg] = useState("");

  const fetchData = useCallback(async () => {
    const headers = getAdminHeaders();
    const [dashRes, resultsRes] = await Promise.all([
      fetch("/api/admin/dashboard", { headers }),
      fetch("/api/admin/results", { headers }),
    ]);
    if (dashRes.ok) setDashboard(await dashRes.json());
    if (resultsRes.ok) {
      const data: ResultsData = await resultsRes.json();
      setAggregated(aggregateResults(data.evaluations));
    }
  }, []);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  async function handleExport() {
    const res = await fetch("/api/admin/export", {
      headers: getAdminHeaders(),
    });
    if (!res.ok) return;
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `ethics-arena-export-${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg("");
    try {
      const text = await file.text();
      const res = await fetch("/api/admin/upload", {
        method: "POST",
        headers: {
          ...getAdminHeaders(),
          "Content-Type": "application/json",
        },
        body: text,
      });
      const data = await res.json();
      setUploadMsg(
        res.ok
          ? `Imported: ${data.dilemmas_created ?? 0} dilemmas, ${data.responses_created ?? 0} responses`
          : `Error: ${data.error}`
      );
      if (res.ok) fetchData();
    } catch (err) {
      console.error("File upload failed:", err);
      setUploadMsg("Upload failed");
    } finally {
      setUploading(false);
    }
  }

  const coverageDisplay = dashboard?.dilemma_coverage
    ? `${dashboard.dilemma_coverage.covered}/${dashboard.dilemma_coverage.total}`
    : "—";

  return (
    <div className="space-y-8">
      <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
        {[
          { label: "Total Evaluations", value: dashboard?.total_evaluations ?? "—" },
          { label: "Unique Evaluators", value: dashboard?.unique_evaluators ?? "—" },
          { label: "Dilemmas Covered", value: coverageDisplay },
          { label: "Avg Time / Eval", value: dashboard?.avg_time ? `${Math.round(dashboard.avg_time)}s` : "—" },
        ].map((card) => (
          <div key={card.label} className="rounded-lg border border-gray-800 bg-gray-900 p-4">
            <div className="text-2xl font-bold text-indigo-400">{card.value}</div>
            <div className="text-xs text-gray-500">{card.label}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-6">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-500">Win Counts</h2>
          {aggregated?.wins && Object.keys(aggregated.wins).length > 0 ? (
            <WinBars wins={aggregated.wins} />
          ) : (
            <p className="text-sm text-gray-600">No evaluations yet</p>
          )}
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-6">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-500">Elo Ratings</h2>
          {aggregated?.elo_ratings && Object.keys(aggregated.elo_ratings).length > 0 ? (
            <EloTable ratings={aggregated.elo_ratings} />
          ) : (
            <p className="text-sm text-gray-600">No evaluations yet</p>
          )}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-6">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-500">Export</h2>
          <p className="mb-3 text-sm text-gray-400">Download all evaluations as pipeline-compatible JSON</p>
          <button onClick={handleExport} className="rounded-lg bg-indigo-600 px-4 py-2 text-sm font-semibold text-white hover:bg-indigo-500">
            Export JSON
          </button>
        </div>
        <div className="rounded-lg border border-gray-800 bg-gray-900 p-6">
          <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-gray-500">Import</h2>
          <p className="mb-3 text-sm text-gray-400">Upload dilemma sets and model responses</p>
          <input type="file" accept=".json" onChange={handleUpload} disabled={uploading} className="text-sm text-gray-400 file:mr-3 file:rounded file:border-0 file:bg-gray-800 file:px-3 file:py-1.5 file:text-sm file:text-gray-300 hover:file:bg-gray-700" />
          {uploadMsg && <p className="mt-2 text-sm text-gray-400">{uploadMsg}</p>}
        </div>
      </div>
    </div>
  );
}
