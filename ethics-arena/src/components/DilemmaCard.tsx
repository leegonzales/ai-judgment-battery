interface DilemmaCardProps {
    title: string;
    scenario: string;
    question: string;
}

export default function DilemmaCard({
    title,
    scenario,
    question,
}: DilemmaCardProps) {
    return (
        <div className="rounded-xl border border-gray-700 bg-gray-900 p-6">
            <h2 className="mb-3 text-xl font-semibold text-gray-100">
                {title}
            </h2>
            <p className="mb-4 whitespace-pre-wrap text-gray-300 leading-relaxed">
                {scenario}
            </p>
            <div className="rounded-lg border border-amber-800/50 bg-amber-950/30 p-4">
                <p className="text-sm font-medium text-amber-400">Question</p>
                <p className="mt-1 text-gray-200">{question}</p>
            </div>
        </div>
    );
}
