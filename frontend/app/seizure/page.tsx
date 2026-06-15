import AppShell from "@/components/AppShell";
import SeizureDashboard, {
  type SeizureStatus,
} from "@/components/SeizureDashboard";
import { apiFetch } from "@/app/lib/api";

export default async function SeizurePage() {
  const status = await apiFetch<SeizureStatus>("/api/seizure/status");

  return (
    <AppShell
      activePath="/seizure"
      title="EEG Seizure Detection"
      subtitle="Synthetic and CHB-MIT EEG seizure analysis"
    >
      {status ? (
        <SeizureDashboard status={status} />
      ) : (
        <BackendError />
      )}
    </AppShell>
  );
}

function BackendError() {
  return (
    <section className="rounded-2xl border border-rose-200 bg-rose-50 p-6">
      <h2 className="text-lg font-semibold text-rose-800">
        Could not load seizure data
      </h2>

      <p className="mt-2 text-sm leading-6 text-rose-700">
        Make sure the FastAPI backend is running and that
        <code className="mx-1 rounded bg-rose-100 px-1.5 py-0.5">
          /api/seizure/status
        </code>
        returns JSON.
      </p>

      <pre className="mt-4 overflow-x-auto rounded-xl bg-slate-950 p-4 text-sm text-slate-100">
        uvicorn backend.main:app --reload
      </pre>
    </section>
  );
}