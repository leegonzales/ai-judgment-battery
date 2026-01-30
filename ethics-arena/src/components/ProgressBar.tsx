interface ProgressBarProps {
    completed: number;
    total: number;
}

export default function ProgressBar({ completed, total }: ProgressBarProps) {
    const pct = total > 0 ? Math.round((completed / total) * 100) : 0;

    return (
        <div className="flex items-center gap-3">
            <span className="text-sm text-gray-400">
                {completed} of {total} completed
            </span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-gray-800">
                <div
                    className="h-full rounded-full bg-indigo-500 transition-all"
                    style={{ width: `${pct}%` }}
                />
            </div>
        </div>
    );
}
