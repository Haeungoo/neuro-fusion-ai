"use client";

import { useEffect, useMemo, useState } from "react";
import { apiFetch, mediaUrl } from "@/app/lib/api";

type FileInfo = {
  exists: boolean;
  path?: string;
  url: string | null;
};

type MotorMetrics = {
  task?: string;
  label_0?: string;
  label_1?: string;
  accuracy?: number;
  precision?: number;
  recall?: number;
  sensitivity?: number;
  specificity?: number;
  f1_score?: number;
  balanced_accuracy?: number;
  true_positive?: number;
  true_negative?: number;
  false_positive?: number;
  false_negative?: number;
  confusion_matrix?: number[][];
  num_trials?: number;
  num_left_hand_trials?: number;
  num_right_hand_trials?: number;
  model?: string;
  dataset?: string;
  num_csp_components?: number;
  num_channels?: number;
  num_samples_per_trial?: number;
  num_train_trials?: number;
  num_test_trials?: number;
};

type MotorStatusResponse = {
  module?: string;
  status?: string;
  description?: string;
  dataset_note?: string;
  labels?: Record<string, string>;
  pipeline?: string[];
  metrics?: MotorMetrics | null;
  outputs?: {
    metrics_json?: FileInfo;
    predictions_csv?: FileInfo;
    confusion_matrix?: FileInfo;
    model?: FileInfo;
    model_file?: FileInfo;
  };
};

function formatPercent(value?: number): string {
  if (value === undefined || value === null) {
    return "N/A";
  }

  return `${(value * 100).toFixed(1)}%`;
}

function formatInteger(value?: number): string {
  if (value === undefined || value === null) {
    return "N/A";
  }

  return String(value);
}

function MetricCard({
  label,
  value,
  description,
  emphasis = false,
}: {
  label: string;
  value: string;
  description: string;
  emphasis?: boolean;
}) {
  return (
    <div
      className={[
        "rounded-2xl border p-4 shadow-sm",
        emphasis
          ? "border-blue-200 bg-blue-50"
          : "border-slate-200 bg-white",
      ].join(" ")}
    >
      <p className="text-sm text-slate-500">{label}</p>

      <p
        className={[
          "mt-2 text-2xl font-semibold",
          emphasis ? "text-blue-700" : "text-slate-900",
        ].join(" ")}
      >
        {value}
      </p>

      <p className="mt-2 text-xs leading-5 text-slate-500">
        {description}
      </p>
    </div>
  );
}

function SectionHeader({
  label,
  title,
  description,
}: {
  label: string;
  title: string;
  description: string;
}) {
  return (
    <div className="mb-5">
      <p className="text-sm font-medium uppercase tracking-wide text-blue-600">
        {label}
      </p>

      <h2 className="mt-1 text-2xl font-semibold text-slate-950">
        {title}
      </h2>

      <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-500">
        {description}
      </p>
    </div>
  );
}

export default function MotorImageryDashboard() {
  const [status, setStatus] =
    useState<MotorStatusResponse | null>(null);

  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadStatus() {
      setLoading(true);

      const data = await apiFetch<MotorStatusResponse>(
        "/api/motor/status",
      );

      setStatus(data);
      setLoading(false);
    }

    loadStatus();
  }, []);

  const metrics = status?.metrics;

  const confusionMatrixUrl = useMemo(() => {
    const url = status?.outputs?.confusion_matrix?.url;

    if (!url) {
      return null;
    }

    return mediaUrl(url);
  }, [status]);

  const modelOutput =
    status?.outputs?.model_file ?? status?.outputs?.model;

  return (
    <main className="min-h-screen bg-slate-50 px-6 py-8 text-slate-900">
      <div className="mx-auto max-w-7xl">
        <section className="rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-center lg:justify-between">
            <div>
              <p className="text-sm font-medium uppercase tracking-wide text-blue-600">
                NeuroFusion-AI
              </p>

              <h1 className="mt-2 text-3xl font-bold text-slate-950">
                EEG Motor Imagery BCI
              </h1>

              <p className="mt-3 max-w-3xl text-sm leading-6 text-slate-600">
                CSP + LDA baseline for classifying left-hand versus
                right-hand motor imagery EEG trials.
              </p>

              {status?.dataset_note && (
                <p className="mt-3 max-w-3xl rounded-2xl bg-amber-50 px-4 py-3 text-sm leading-6 text-amber-800">
                  {status.dataset_note}
                </p>
              )}
            </div>

            <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3">
              <p className="text-sm text-slate-500">
                Backend status
              </p>

              <p className="mt-1 text-lg font-semibold text-slate-900">
                {loading
                  ? "Loading..."
                  : status?.status ?? "Unavailable"}
              </p>
            </div>
          </div>
        </section>

        <section className="mt-8 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <SectionHeader
            label="Step 1"
            title="Final Motor Imagery Evaluation"
            description="These metrics come from the CSP + LDA motor imagery result file."
          />

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="Accuracy"
              value={formatPercent(metrics?.accuracy)}
              description="Overall proportion of correctly classified trials."
              emphasis
            />

            <MetricCard
              label="Precision"
              value={formatPercent(metrics?.precision)}
              description="Among predicted right-hand trials, how many were truly right-hand imagery."
            />

            <MetricCard
              label="Recall"
              value={formatPercent(metrics?.recall)}
              description="How well the model detects true right-hand imagery trials."
            />

            <MetricCard
              label="F1 score"
              value={formatPercent(metrics?.f1_score)}
              description="Balance between precision and recall."
              emphasis
            />

            <MetricCard
              label="Specificity"
              value={formatPercent(metrics?.specificity)}
              description="How well the model identifies left-hand imagery trials."
            />

            <MetricCard
              label="Balanced accuracy"
              value={formatPercent(metrics?.balanced_accuracy)}
              description="Average of sensitivity and specificity."
            />

            <MetricCard
              label="Total trials"
              value={formatInteger(metrics?.num_trials)}
              description="Number of test trials used for evaluation."
            />

            <MetricCard
              label="CSP components"
              value={formatInteger(metrics?.num_csp_components)}
              description="Number of CSP spatial filters used."
            />
          </div>

          {confusionMatrixUrl ? (
            <div className="mt-8 rounded-2xl border border-slate-200 bg-slate-50 p-4">
              <h3 className="text-lg font-semibold text-slate-900">
                Confusion Matrix
              </h3>

              <p className="mt-1 text-sm text-slate-500">
                Left-hand imagery and right-hand imagery
                classification result.
              </p>

              <img
                src={confusionMatrixUrl}
                alt="Motor imagery confusion matrix"
                className="mt-4 w-full rounded-xl border border-slate-100 bg-white"
              />
            </div>
          ) : (
            <div className="mt-8 rounded-2xl border border-amber-200 bg-amber-50 p-4">
              <h3 className="text-lg font-semibold text-amber-900">
                Confusion matrix not found
              </h3>

              <p className="mt-1 text-sm leading-6 text-amber-800">
                Run{" "}
                <code className="rounded bg-white px-1 py-0.5">
                  python -m scripts.train_motor_imagery_csp_lda
                </code>{" "}
                to generate the motor imagery result files.
              </p>
            </div>
          )}
        </section>

        <section className="mt-8 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <SectionHeader
            label="Step 2"
            title="CSP + LDA Pipeline"
            description="CSP extracts spatial EEG patterns that separate left-hand and right-hand motor imagery. LDA then classifies those features."
          />

          <div className="grid gap-3 md:grid-cols-5">
            {(status?.pipeline ?? [
              "EEG trials",
              "CSP spatial filtering",
              "Log-variance features",
              "Linear Discriminant Analysis",
              "Left/right imagery prediction",
            ]).map((step, index) => (
              <div
                key={step}
                className="rounded-2xl border border-slate-200 bg-slate-50 p-4"
              >
                <p className="text-xs font-medium uppercase tracking-wide text-blue-600">
                  {`0${index + 1}`}
                </p>

                <p className="mt-2 text-sm font-semibold text-slate-900">
                  {step}
                </p>
              </div>
            ))}
          </div>
        </section>

        <section className="mt-8 rounded-3xl border border-slate-200 bg-white p-6 shadow-sm">
          <SectionHeader
            label="Step 3"
            title="Technical Details"
            description="These values help debug and understand how the final metrics were calculated."
          />

          <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
            <MetricCard
              label="True positive"
              value={formatInteger(metrics?.true_positive)}
              description="Right-hand imagery trials correctly classified."
            />

            <MetricCard
              label="True negative"
              value={formatInteger(metrics?.true_negative)}
              description="Left-hand imagery trials correctly classified."
            />

            <MetricCard
              label="False positive"
              value={formatInteger(metrics?.false_positive)}
              description="Left-hand imagery trials incorrectly classified as right-hand."
            />

            <MetricCard
              label="False negative"
              value={formatInteger(metrics?.false_negative)}
              description="Right-hand imagery trials incorrectly classified as left-hand."
            />

            <MetricCard
              label="Model"
              value={metrics?.model ?? "CSP + LDA"}
              description="Model architecture used for this baseline."
            />

            <MetricCard
              label="Dataset"
              value={metrics?.dataset ?? "N/A"}
              description="Dataset used to generate the current result."
            />

            <MetricCard
              label="Model file"
              value={modelOutput?.exists ? "Saved" : "Missing"}
              description="Whether the trained CSP + LDA model file exists."
            />

            <MetricCard
              label="Predictions CSV"
              value={
                status?.outputs?.predictions_csv?.exists
                  ? "Saved"
                  : "Missing"
              }
              description="Trial-level prediction CSV output."
            />
          </div>
        </section>
      </div>
    </main>
  );
}