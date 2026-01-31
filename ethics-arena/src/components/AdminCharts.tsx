"use client";

interface WinBarProps {
  wins: Record<string, number>;
}

export function WinBars({ wins }: WinBarProps) {
  const max = Math.max(...Object.values(wins), 1);
  const sorted = Object.entries(wins).sort(([, a], [, b]) => b - a);

  return (
    <div className="space-y-3">
      {sorted.map(([model, count]) => (
        <div key={model}>
          <div className="mb-1 flex justify-between text-sm">
            <span className="text-gray-300">{model}</span>
            <span className="text-gray-500">{count} wins</span>
          </div>
          <div className="h-3 w-full rounded-full bg-gray-800">
            <div
              className="h-3 rounded-full bg-indigo-500"
              style={{ width: `${(count / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

interface EloTableProps {
  ratings: Record<string, number>;
}

export function EloTable({ ratings }: EloTableProps) {
  const sorted = Object.entries(ratings).sort(([, a], [, b]) => b - a);

  function badge(rating: number) {
    if (rating >= 1550) return "bg-green-900 text-green-300";
    if (rating >= 1480) return "bg-yellow-900 text-yellow-300";
    return "bg-red-900 text-red-300";
  }

  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="border-b border-gray-800 text-left text-gray-500">
          <th className="pb-2">Rank</th>
          <th className="pb-2">Model</th>
          <th className="pb-2 text-right">Elo</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map(([model, elo], i) => (
          <tr key={model} className="border-b border-gray-800/50">
            <td className="py-2 text-gray-500">#{i + 1}</td>
            <td className="py-2 text-gray-300">{model}</td>
            <td className="py-2 text-right">
              <span
                className={`rounded px-2 py-0.5 text-xs font-semibold ${badge(elo)}`}
              >
                {elo}
              </span>
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
