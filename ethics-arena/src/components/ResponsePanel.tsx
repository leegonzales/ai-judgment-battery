interface ResponsePanelProps {
    label: string;
    text: string;
    rank: number | null;
    onClick: () => void;
}

const rankBadge: Record<number, { text: string; color: string }> = {
    1: { text: "1st", color: "bg-yellow-500 text-black" },
    2: { text: "2nd", color: "bg-gray-400 text-black" },
    3: { text: "3rd", color: "bg-amber-700 text-white" },
};

export default function ResponsePanel({
    label,
    text,
    rank,
    onClick,
}: ResponsePanelProps) {
    const badge = rank ? rankBadge[rank] : null;
    const borderColor = rank
        ? "border-indigo-500"
        : "border-gray-700 hover:border-gray-500";

    return (
        <button
            type="button"
            onClick={onClick}
            className={`w-full cursor-pointer rounded-xl border-2 ${borderColor} bg-gray-900 p-5 text-left transition-colors`}
        >
            <div className="mb-3 flex items-center justify-between">
                <span className="text-sm font-semibold text-gray-400">
                    Response {label}
                </span>
                {badge && (
                    <span
                        className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${badge.color}`}
                    >
                        {badge.text}
                    </span>
                )}
            </div>
            <div className="max-h-72 overflow-y-auto pr-2">
                <p className="whitespace-pre-wrap text-gray-300 leading-relaxed">
                    {text}
                </p>
            </div>
        </button>
    );
}
