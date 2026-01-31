"use client";

import { useState, useEffect, ReactNode } from "react";

export default function AdminLayout({ children }: { children: ReactNode }) {
  const [password, setPassword] = useState("");
  const [authed, setAuthed] = useState(false);
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(true);

  useEffect(() => {
    const stored = sessionStorage.getItem("admin_password");
    if (stored) {
      verifyPassword(stored);
    } else {
      setChecking(false);
    }
  }, []);

  async function verifyPassword(pw: string) {
    setChecking(true);
    setError("");
    try {
      const res = await fetch("/api/admin/dashboard", {
        headers: { Authorization: `Bearer ${pw}` },
      });
      if (res.ok) {
        sessionStorage.setItem("admin_password", pw);
        setAuthed(true);
      } else {
        sessionStorage.removeItem("admin_password");
        setError("Invalid password");
      }
    } catch {
      setError("Connection error");
    } finally {
      setChecking(false);
    }
  }

  function handleLogout() {
    sessionStorage.removeItem("admin_password");
    setAuthed(false);
    setPassword("");
  }

  if (checking) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-950">
        <p className="text-gray-500">Checking authentication...</p>
      </div>
    );
  }

  if (!authed) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-gray-950 px-4">
        <div className="w-full max-w-sm">
          <h1 className="mb-6 text-center text-2xl font-bold text-gray-100">
            Admin Login
          </h1>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              verifyPassword(password);
            }}
          >
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="Admin password"
              className="mb-3 w-full rounded-lg border border-gray-700 bg-gray-900 px-4 py-3 text-gray-100 placeholder-gray-600 focus:border-indigo-500 focus:outline-none"
              autoFocus
            />
            {error && (
              <p className="mb-3 text-sm text-red-400">{error}</p>
            )}
            <button
              type="submit"
              className="w-full rounded-lg bg-indigo-600 px-4 py-3 text-sm font-semibold text-white hover:bg-indigo-500"
            >
              Login
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-950">
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="mx-auto flex max-w-6xl items-center justify-between">
          <h1 className="text-lg font-bold text-gray-100">
            Ethics<span className="text-indigo-400">Arena</span>{" "}
            <span className="text-sm font-normal text-gray-500">Admin</span>
          </h1>
          <button
            onClick={handleLogout}
            className="text-sm text-gray-500 hover:text-gray-300"
          >
            Logout
          </button>
        </div>
      </header>
      <main className="mx-auto max-w-6xl px-6 py-8">{children}</main>
    </div>
  );
}
