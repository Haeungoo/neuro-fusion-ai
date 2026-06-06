import { BrainCircuit, Database, Gauge, Waves } from "lucide-react";

import AppShell from "@/components/AppShell";
import StatCard from "@/components/StatCard";
import StatusBadge from "@/components/StatusBadge";
import { apiFetch, mediaUrl } from "@/lib/api";

type FileInfo = {
  path: string;
  exists: boolean;
};

type MotorStatus = {
  module: string;
  dataset: string;
  model: string;
  pipeline: string[];
  model_file: FileInfo;
  outputs: {
    confusion_matrix: FileInfo;
    csp_patterns: FileInfo;
    metrics: FileInfo;
  };
  metrics: {
    accuracy?: number;
    f1_score?: number;
    n_epochs?: number;
    n_channels?: number;
    sfreq?: number;
  };
  note: string;
};

export default async function MotorPage() {
  const status = await apiFetch<MotorStatus>("/api/motor/status");

  return (
    <AppShell
      activePath="/motor"
      title="EEG Motor Imagery BCI"
      subtitle="PhysioNet EEGBCI classification using CSP and LDA"
    >
      <section className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          icon={BrainCircuit}
          label="Model"
          value={status?.model ?? "CSP + LDA"}
          description={status?.model_file.exists ? "Available" : "Missing"}
          iconClassName="bg-emerald-50 text-emerald-600"
        />
        <StatCard
          icon={Database}
          label="Dataset"
          value="EEGBCI"
          description="PhysioNet"
          iconClassName="bg-blue-50 text-blue-600"
        />
        <StatCard
          icon={Gauge}
          label="Accuracy"
          value={formatMetric(status?.metrics?.accuracy)}
          description="Classification"
          iconClassName="bg-violet-50 text-violet-600"
        />
        <StatCard
          icon={Waves}
          label="F1 score"
          value={formatMetric(status?.metrics?.f1_score)}
          description="Weighted"
        />
      </section>

      <section className="mt-5 grid gap-5 xl:grid-cols-2">
        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h2 className="text-lg font-semibold text-slate-900">
                CSP spatial patterns
              </h2>
              <p className="mt-1 text-sm text-slate-500">
                Spatial EEG patterns used to discriminate imagery classes.
              </p>
            </div>
            <StatusBadge active={status?.outputs.csp_patterns.exists ?? false} />
          </div>

          <div className="mt-5 flex min-h-80 items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-white p-3">
            <img
              src={mediaUrl("motor/csp_patterns.png")}
              alt="CSP spatial patterns"
              className="max-h-[440px] w-full object-contain"
            />
          </div>
        </article>

        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <div>
            <h2 className="text-lg font-semibold text-slate-900">
              Classification performance
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Confusion matrix for the CSP and LDA classifier.
            </p>
          </div>

          <div className="mt-5 flex min-h-80 items-center justify-center overflow-hidden rounded-xl border border-slate-200 bg-white p-3">
            <img
              src={mediaUrl("motor/confusion_matrix_motor.png")}
              alt="Motor imagery confusion matrix"
              className="max-h-[440px] w-full object-contain"
            />
          </div>
        </article>
      </section>

      <section className="mt-5 grid gap-5 lg:grid-cols-[1fr_0.7fr]">
        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">
            Processing pipeline
          </h2>

          <div className="mt-5 grid gap-3 sm:grid-cols-2">
            {(status?.pipeline ?? []).map((step, index) => (
              <div
                key={step}
                className="flex items-start gap-3 rounded-xl bg-slate-50 p-4"
              >
                <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-teal-600 text-xs font-bold text-white">
                  {index + 1}
                </span>
                <p className="text-sm font-medium leading-5 text-slate-700">
                  {step}
                </p>
              </div>
            ))}
          </div>
        </article>

        <article className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="text-lg font-semibold text-slate-900">
            Dataset summary
          </h2>

          <dl className="mt-4 divide-y divide-slate-100">
            <MetricRow
              label="Epochs"
              value={formatNumber(status?.metrics?.n_epochs)}
            />
            <MetricRow
              label="Channels"
              value={formatNumber(status?.metrics?.n_channels)}
            />
            <MetricRow
              label="Sampling rate"
              value={
                status?.metrics?.sfreq == null
                  ? "N/A"
                  : `${status.metrics.sfreq} Hz`
              }
            />
          </dl>
        </article>
      </section>

      <section className="mt-5 rounded-xl border border-blue-200 bg-blue-50 p-4 text-sm text-blue-800">
        {status?.note ??
          "CSP patterns are discriminative spatial patterns, not direct activation maps."}
      </section>
    </AppShell>
  );
}

function MetricRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4 py-3">
      <dt className="text-sm text-slate-500">{label}</dt>
      <dd className="text-sm font-semibold text-slate-800">{value}</dd>
    </div>
  );
}

function formatMetric(value?: number): string {
  return value == null ? "N/A" : value.toFixed(3);
}

function formatNumber(value?: number): string {
  return value == null ? "N/A" : value.toLocaleString();
}