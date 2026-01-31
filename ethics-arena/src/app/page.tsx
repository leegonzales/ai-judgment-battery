import Link from "next/link";

export default function Home() {
    return (
        <div className="flex min-h-screen flex-col items-center justify-center bg-gray-950 px-6 text-white">
            <main className="flex max-w-2xl flex-col items-center gap-8 text-center">
                <h1 className="text-5xl font-bold tracking-tight sm:text-6xl">
                    Ethics<span className="text-indigo-400">Arena</span>
                </h1>

                <p className="text-lg text-gray-400">
                    Blind human evaluation of AI ethical reasoning
                </p>

                <div className="max-w-xl space-y-4 text-base leading-relaxed text-gray-300">
                    <p>
                        You will read real ethical dilemmas and rank three
                        AI-generated responses from best to worst. Model
                        identities are hidden so your judgment stays
                        unbiased.
                    </p>
                    <p>
                        Each evaluation takes about 2 minutes. Your
                        rankings feed an Elo rating system that surfaces
                        which AI reasons most like a careful human.
                    </p>
                </div>

                <Link
                    href="/evaluate"
                    className="mt-4 rounded-lg bg-indigo-600 px-8 py-3 text-lg font-semibold transition-colors hover:bg-indigo-500"
                >
                    Start Evaluating
                </Link>
            </main>

            <footer className="absolute bottom-6 text-sm text-gray-600">
                <Link
                    href="/admin"
                    className="transition-colors hover:text-gray-400"
                >
                    Admin
                </Link>
            </footer>
        </div>
    );
}
